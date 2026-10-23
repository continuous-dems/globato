"""Globato-only module preparation for Stack QC; Fetchez is unchanged.

Ordinary recipes delegate to Fetchez's own PipelinePlan. A recipe with a
StackQC hook prepares selected source collections once, then hands accepted
FusionState streams to the standard Fetchez/Globato source consumers.
"""

import logging
import tracemalloc

from fetchez.recipe import Recipe, PipelinePlan
from fetchez.core import run_fetchez
from fetchez.utils import _log_hook_history

POSITION = "_stack_qc_position"

logger = logging.getLogger(__name__)


class PreparedSources:
    """In-memory delivery view; NOT a Fetchez lifecycle API or upstream patch."""

    def __init__(self, original, entries):
        self.original = original
        self.results = list(entries)
        self.hooks = []
        for entry in self.results:
            if not isinstance(entry, dict):
                raise TypeError("Stack QC accepted entries must be dictionaries")
            if entry.get("stream") is None:
                # Existing Fetchez auto-initializes empty streams in normal modules.
                # A no-data ancillary entry must not trigger a second raw read.
                entry["stream"] = iter(())
                entry["stream_type"] = "raster-stream"

    def __getattr__(self, key):
        return getattr(object.__getattribute__(self, "original"), key)

    def fetch_entry(self, entry, **kwargs):
        return 0  # Source was already fetched, transformed, and verified.


class StreamGate:
    """Skip original source transforms on already prepared source streams."""

    def __init__(self, hook):
        self.hook = hook
        self.name = hook.name
        self.stage = hook.stage
        self.closed = False

    def __getattr__(self, key):
        return getattr(object.__getattribute__(self, "hook"), key)

    def run(self, entries):
        fresh = [
            (owner, entry)
            for owner, entry in entries
            if not isinstance(owner, PreparedSources)
        ]
        if not fresh:
            return entries
        result = self.hook.run(fresh)
        if result is None:
            result = []
        prepared_present = len(fresh) != len(entries)
        if not prepared_present:
            return result
        if len(result) != len(fresh):
            # Fetchez runs hooks on one source at a time. A mixed batch that
            # changes cardinality has no unambiguous ordering, so fail rather
            # than quietly moving prepared contributions to the end.
            raise ValueError(
                f"{self.name} changed the number of ordinary entries in a "
                "mixed prepared/ordinary batch"
            )
        replacement = iter(result)
        return [
            item if isinstance(item[0], PreparedSources) else next(replacement)
            for item in entries
        ]

    def teardown(self):
        # Runtime teardown must reach the underlying hook exactly once.
        if not self.closed:
            self.closed = True
            teardown = getattr(self.hook, "teardown", None)
            if callable(teardown):
                teardown()


class DeliveryOrder:
    """Restore the selected manifest order before native fetch submission.

    Fetchez flattens modules before its global manifest stage. Preparation must
    not undo an earlier manifest's cross-module ordering when regrouping sources.
    Fetch completion still follows stock Fetchez's concurrent execution policy.
    """

    name = "stack_qc_delivery_order"
    stage = "manifest"

    def run(self, entries):
        return sorted(entries, key=lambda item: item[1][POSITION])


def _teardown(hooks):
    seen = set()
    for hook in hooks:
        if id(hook) in seen:
            continue
        seen.add(id(hook))
        close = getattr(hook, "teardown", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.exception("Teardown failed for hook %s", hook.name)


def _cache_lookup(hook):
    # The native lookup is intentionally delayed until ALL source-selection
    # rules have run. No generic deferred lifecycle is added to Fetchez.
    from globato.hooks.sinks.side_stack import SideStackCheck

    return isinstance(hook, SideStackCheck)


def _qc_owner(module):
    owners = [h for h in module.hooks if getattr(h, "name", "") == "stack_qc"]
    if len(owners) > 1:
        raise ValueError(f"Module {module.name!r} has multiple Stack QC hooks")
    return owners[0] if owners else None


class StackQCPlan(PipelinePlan):
    """Apply ordinary source selection once, then run native Fetchez consumers."""

    def execute(self):
        if self.ignore_failures:
            raise ValueError("Stack QC requires --fail-fast (ignore_failures=False)")
        tracing_started = not tracemalloc.is_tracing()
        if tracing_started:
            tracemalloc.start()
        self._native_started = False
        self._native_hook_ids = set()
        # Review publishers are temporary runtime hooks, appended during source
        # preparation. Do not leave them on a reusable compiled plan, including
        # when preparation or the native build fails.
        original_global_hooks = list(self.global_hooks)
        try:
            return self._execute_prepared()
        finally:
            # Native Fetchez owns teardown for consumers it receives. Selection
            # hooks and QC module hooks are removed before handoff; close those
            # here, including discovery/selection/preparation error paths.
            leftovers = [h for h in self.global_hooks if h.stage == "manifest"]
            for module in self.modules:
                leftovers.extend(
                    h
                    for h in module.hooks
                    if h.stage == "manifest"
                    or any(x.name == "stack_qc" for x in module.hooks)
                )
            if not self._native_started:
                leftovers.extend(self.global_hooks)
                leftovers.extend(h for module in self.modules for h in module.hooks)
            if self._native_started:
                leftovers = [h for h in leftovers if id(h) not in self._native_hook_ids]
            _teardown(leftovers)
            self.global_hooks[:] = original_global_hooks
            if tracing_started:
                current, peak = tracemalloc.get_traced_memory()
                logger.info(
                    "Pipeline Memory Usage - Current: %.2f MB | Peak: %.2f MB",
                    current / 10**6,
                    peak / 10**6,
                )
                tracemalloc.stop()

    def _select_sources(self):
        """Match Fetchez's discovery → module manifests → global manifests."""
        # Resolve owners once; duplicate QC attachments fail before provider APIs
        # are contacted, and all manifest stages use the same ownership decision.
        owners = {id(module): _qc_owner(module) for module in self.modules}
        for module in self.modules:
            for hook in module.hooks:
                if (
                    hook.stage == "manifest"
                    and owners[id(module)]
                    and getattr(hook, "defer_to_preparation", False)
                    and not _cache_lookup(hook)
                ):
                    raise ValueError(
                        f"{hook.name}: deferred module manifest hooks are not "
                        "supported by Globato-only Stack QC preparation"
                    )
        for hook in self.global_hooks:
            if (
                hook.stage == "manifest"
                and getattr(hook, "defer_to_preparation", False)
                and not _cache_lookup(hook)
            ):
                raise ValueError(
                    f"{hook.name}: deferred global manifest hooks are not "
                    "supported by Globato-only Stack QC preparation"
                )
        for module in self.modules:
            module.run()
        for module in self.modules:
            selected = [(module, entry) for entry in module.results]
            for hook in module.hooks:
                if hook.stage != "manifest":
                    continue
                if owners[id(module)] and _cache_lookup(hook):
                    continue
                selected = hook.run(selected)
                if selected is None:
                    selected = []
                _log_hook_history(selected, hook)
            module.results = [entry for _, entry in selected]

        selected = [
            (module, entry) for module in self.modules for entry in module.results
        ]
        for hook in self.global_hooks:
            if hook.stage != "manifest":
                continue
            if _cache_lookup(hook):
                ordinary = [(m, e) for m, e in selected if not owners.get(id(m))]
                if ordinary:
                    hook.run(ordinary)
                    _log_hook_history(ordinary, hook)
                continue
            updated = hook.run(selected)
            if isinstance(updated, list):
                selected = updated
            _log_hook_history(selected, hook)

        grouped = {id(module): [] for module in self.modules}
        for position, (module, entry) in enumerate(selected):
            entry[POSITION] = position
            if id(module) not in grouped:
                raise ValueError("A manifest hook introduced an unselected module")
            grouped[id(module)].append(entry)
        for module in self.modules:
            module.results = grouped[id(module)]

    def _execute_prepared(self):
        from .preparation import SOURCE_CONSUMERS

        # Explicit QC requests are configuration, not provider-dependent work.
        # Validate before discovery so API outages cannot hide a bad grid request.
        from .resampling import validate_request

        for module in self.modules:
            owner = _qc_owner(module)
            if owner and getattr(owner, "qc_res", None) is not None:
                validate_request(owner.qc_res, module.region, self.global_hooks)
        self._select_sources()
        modules = []
        for module in self.modules:
            owner = _qc_owner(module)
            if owner:
                modules.append(
                    owner.prepare_module(
                        module, self.global_hooks, threads=self.threads
                    )
                )
            else:
                modules.append(module)

        self._native_hook_ids = {
            id(h) for h in self.global_hooks if h.stage != "manifest"
        }
        self._native_hook_ids.update(
            id(h)
            for module in self.modules
            if not _qc_owner(module)
            for h in module.hooks
            if h.stage != "manifest"
        )
        for module in modules:
            if any(POSITION not in entry for entry in module.results):
                raise ValueError(
                    "Source preparation discarded the selected source position"
                )

        # Fetchez replays module manifests when consuming a collection.
        # Remove those already run above, then restore them on every exit.
        removals = []
        for module in self.modules:
            for attr in ("internal_hooks", "external_hooks"):
                original = getattr(module, attr, None)
                if original is not None:
                    removals.append((module, attr, original))
                    setattr(
                        module, attr, [h for h in original if h.stage != "manifest"]
                    )
        resumes = []
        try:
            hooks = [DeliveryOrder()]
            gates = {}
            for hook in self.global_hooks:
                if hook.stage == "manifest":
                    continue
                if hook.stage == "file" or (
                    hook.stage == "stream"
                    and hook.name.replace("-", "_") not in SOURCE_CONSUMERS
                ):
                    gate = gates.setdefault(id(hook), StreamGate(hook))
                    hooks.append(gate)
                else:
                    hooks.append(hook)
            # Reuse a gate for an object shared across scopes so native teardown
            # sees one identity, just as it does without Stack QC.
            for module in self.modules:
                for attr in ("internal_hooks", "external_hooks"):
                    current = getattr(module, attr, None)
                    if current is not None:
                        setattr(module, attr, [gates.get(id(h), h) for h in current])
            # Final additive states cannot subtract contributions accepted by
            # an older QC mask. Rebuild final consumers; expensive raw sources,
            # side-stacks and scientific contexts still reuse validated caches.
            resumes = [
                (h, h.resume)
                for h in self.global_hooks
                if h.name.replace("-", "_") in SOURCE_CONSUMERS and hasattr(h, "resume")
            ]
            for hook, _ in resumes:
                hook.resume = False
            self._native_started = any(module.results for module in modules)
            return run_fetchez(
                modules, threads=self.threads, global_hooks=hooks, ignore_failures=False
            )
        finally:
            for hook, original in resumes:
                hook.resume = original
            for module, attr, original in removals:
                setattr(module, attr, original)


class StackQCRecipe(Recipe):
    """Default Fetchez Recipe except when a module requests Stack QC."""

    def compile(self, *args, **kwargs):
        plan = super().compile(*args, **kwargs)
        if any(h.name == "stack_qc" for h in plan.global_hooks):
            raise ValueError("Attach stack_qc to a module, not global_hooks")
        if any(_qc_owner(module) for module in plan.modules):
            return StackQCPlan(**vars(plan))
        return plan
