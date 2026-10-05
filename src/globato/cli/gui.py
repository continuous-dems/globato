# globato/cli/gui.py

import logging
import os
import subprocess
import sys
import threading
import time
import webbrowser
from importlib import resources

import click
from fetchez.utils import FetchezMainCommand

logger = logging.getLogger(__name__)


@click.command("gui", cls=FetchezMainCommand)
def gui_cmd():
    """Launch the experimental interactive Globato notebook."""

    click.secho("Starting Globato Web Server...", fg="green", bold=True)
    click.secho(
        "A browser window will open shortly at http://localhost:8866/", fg="cyan"
    )
    click.secho("Press Ctrl+C in this terminal to stop the server.", fg="yellow")

    temp_nb_name = "globato_gui_temp.ipynb"

    def _open_browser():
        # Give Voilà a second to initialize before opening the browser directly
        time.sleep(1.5)
        webbrowser.open("http://localhost:8866/")

    try:
        nb_content = (
            resources.files("globato.gui").joinpath("globato_app.ipynb").read_text()
        )
        with open(temp_nb_name, "w", encoding="utf-8") as f:
            f.write(nb_content)

        threading.Thread(target=_open_browser, daemon=True).start()

        subprocess.run(
            [
                sys.executable,
                "-m",
                "voila",
                temp_nb_name,
                "--theme=light",
                "--no-browser",
            ],
            check=True,
        )

    except subprocess.CalledProcessError:
        click.secho("Error: The web server crashed.", fg="red", bold=True)
        sys.exit(1)
    except FileNotFoundError:
        click.secho(
            "Error: 'voila' is not installed. Run `pip install voila`", fg="red"
        )
        sys.exit(1)
    except Exception as e:
        click.secho(f"Failed to launch GUI: {e}", fg="red")
        sys.exit(1)
    finally:
        if os.path.exists(temp_nb_name):
            os.remove(temp_nb_name)
