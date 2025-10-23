#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import importlib


try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    USE_RICH = True
except ImportError:
    
    print("Warning: 'rich' library not found, falling back to plain text.")
    print("You can install it with: pip install rich")
    USE_RICH = False
    
    class Console:
        def print(self, *args, **kwargs):
            print(*args, **kwargs)
    
    class Table:
        def __init__(self, *args, **kwargs):
            self.title = kwargs.get("title", "")
            self.rows = []
            self.columns = []
            print(f"\n--- {self.title} ---")
        
        def add_column(self, *args, **kwargs):
            self.columns.append(args[0])
        
        def add_row(self, *args):
            self.rows.append(args)
            
        def print_table(self):
          
            print(" | ".join(self.columns))
            print("-" * (sum(len(c) for c in self.columns) + len(self.columns) * 3))
    
            for row in self.rows:
                print(" | ".join(row))

    console = Console()


REQUIRED_LIBRARIES = [
    "jax",
    "rich",
    "psutil"
]


def check_dependencies():
    """
    Checks for required libraries and prints the status.
    """
    if USE_RICH:
        table = Table(title="Dependency Check", header_style="bold magenta")
        table.add_column("Library", style="cyan", width=20)
        table.add_column("Status", style="green")
    else:

        table = Table(title="Dependency Check")
        table.add_column("Library")
        table.add_column("Status")

    missing_libs = []

    console.print(f"Checking {len(REQUIRED_LIBRARIES)} libraries...")

    for lib_name in REQUIRED_LIBRARIES:
        try:

            importlib.import_module(lib_name)
            
            if USE_RICH:
                table.add_row(lib_name, "✅ Installed")
            else:
                table.add_row(lib_name, "Installed")
                
        except ImportError:
      
            missing_libs.append(lib_name)
            if USE_RICH:
                table.add_row(lib_name, "[red]❌ NOT FOUND[/red]")
            else:
                table.add_row(lib_name, "NOT FOUND")


    if USE_RICH:
        console.print(table)
    else:
        table.print_table() 

    console.print("\n" + "-"*30 + "\n") 

    if not missing_libs:
        console.print("[green]✅ Excellent! All required libraries are installed.[/green]" if USE_RICH else "✅ Excellent! All required libraries are installed.")
        return True
    else:

        console.print(f"[red]❌ Error: {len(missing_libs)} missing libraries:[/red] {', '.join(missing_libs)}" if USE_RICH else f"❌ Error: {len(missing_libs)} missing libraries: {', '.join(missing_NextActions)}")
        console.print(f"[yellow]Please install the missing libraries using this command:[/yellow]" if USE_RICH else "Please install the missing libraries using this command:")
        
        sys.exit(1)

if __name__ == "__main__":
    check_dependencies()
