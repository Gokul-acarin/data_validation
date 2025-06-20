#!/usr/bin/env python3
"""
Show project status and statistics.

This script provides information about the current state of the data validation
project, including file counts, recent activity, and configuration status.
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any


def get_directory_stats(base_path: Path) -> Dict[str, Any]:
    """Get statistics about project directories."""
    stats = {}
    
    directories = [
        ("Data Directory", base_path / "data"),
        ("Output Directory", base_path / "output"),
        ("Config Directory", base_path / "config"),
        ("Source Directory", base_path / "src"),
        ("Scripts Directory", base_path / "scripts"),
        ("Tests Directory", base_path / "tests"),
    ]
    
    for name, path in directories:
        if path.exists():
            try:
                # Count files recursively
                file_count = sum(1 for _ in path.rglob("*") if _.is_file())
                dir_count = sum(1 for _ in path.rglob("*") if _.is_dir())
                total_size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                
                stats[f"{name} Files"] = file_count
                stats[f"{name} Subdirs"] = dir_count
                stats[f"{name} Size (MB)"] = round(total_size / (1024 * 1024), 2)
            except Exception as e:
                print(f"Warning: Failed to get stats for {name}: {e}")
                stats[f"{name} Files"] = "Error"
                stats[f"{name} Subdirs"] = "Error"
                stats[f"{name} Size (MB)"] = "Error"
        else:
            stats[f"{name} Files"] = "Not found"
            stats[f"{name} Subdirs"] = "Not found"
            stats[f"{name} Size (MB)"] = "Not found"
    
    return stats


def get_file_stats(base_path: Path) -> Dict[str, Any]:
    """Get statistics about specific file types."""
    stats = {}
    
    # Count files by type
    file_types = {
        "CSV Files": "*.csv",
        "JSON Files": "*.json",
        "YAML Files": "*.yaml",
        "Python Files": "*.py",
        "Text Files": "*.txt",
        "Log Files": "*.log",
        "Shell Scripts": "*.sh",
    }
    
    for file_type, pattern in file_types.items():
        try:
            count = sum(1 for _ in base_path.rglob(pattern))
            stats[file_type] = count
        except Exception as e:
            print(f"Warning: Failed to count {file_type}: {e}")
            stats[file_type] = "Error"
    
    # Count generated data files
    try:
        generated_files = list((base_path / "data").rglob("fake_data_*.csv"))
        stats["Generated Data Files"] = len(generated_files)
        
        anomaly_files = list((base_path / "data").rglob("*_with_anomalies_*.csv"))
        stats["Anomaly Files"] = len(anomaly_files)
        
        mapping_files = list((base_path / "output").rglob("field_*.csv"))
        stats["Mapping Files"] = len(mapping_files)
        
    except Exception as e:
        print(f"Warning: Failed to count specific file types: {e}")
        stats["Generated Data Files"] = "Error"
        stats["Anomaly Files"] = "Error"
        stats["Mapping Files"] = "Error"
    
    return stats


def get_recent_activity(base_path: Path, days: int = 7) -> Dict[str, Any]:
    """Get recent activity statistics."""
    stats = {}
    
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_files = []
        
        for file_path in base_path.rglob("*"):
            if file_path.is_file():
                try:
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if mtime > cutoff_date:
                        recent_files.append((file_path, mtime))
                except Exception:
                    continue
        
        # Sort by modification time
        recent_files.sort(key=lambda x: x[1], reverse=True)
        
        stats[f"Files Modified (Last {days} Days)"] = len(recent_files)
        
        if recent_files:
            latest_file, latest_time = recent_files[0]
            stats["Most Recent File"] = f"{latest_file.name} ({latest_time.strftime('%Y-%m-%d %H:%M')})"
        else:
            stats["Most Recent File"] = "None"
        
        # Count by type
        csv_count = sum(1 for f, _ in recent_files if f.suffix == '.csv')
        py_count = sum(1 for f, _ in recent_files if f.suffix == '.py')
        
        stats[f"CSV Files Modified ({days} Days)"] = csv_count
        stats[f"Python Files Modified ({days} Days)"] = py_count
        
    except Exception as e:
        print(f"Warning: Failed to get recent activity: {e}")
        stats[f"Files Modified (Last {days} Days)"] = "Error"
        stats["Most Recent File"] = "Error"
        stats[f"CSV Files Modified ({days} Days)"] = "Error"
        stats[f"Python Files Modified ({days} Days)"] = "Error"
    
    return stats


def get_config_status(base_path: Path) -> Dict[str, Any]:
    """Get configuration status information."""
    stats = {}
    
    # Check for configuration files
    config_files = [
        "config/validation_rules.txt",
        "config/environments/development.yaml",
        "config/environments/production.yaml",
        "pyproject.toml",
        ".gitignore",
    ]
    
    for config_file in config_files:
        file_path = base_path / config_file
        stats[f"Config: {config_file}"] = "Found" if file_path.exists() else "Missing"
    
    # Check environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    stats["OpenAI API Key"] = "Configured" if openai_key else "Not configured"
    
    # Check Python environment
    stats["Python Version"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    stats["Current Directory"] = str(base_path.absolute())
    
    return stats


def print_status_table(stats: Dict[str, Any], title: str = "Project Status") -> None:
    """Print statistics in a formatted table."""
    print("\n" + "="*60)
    print(title.upper())
    print("="*60)
    
    # Find the longest key for formatting
    max_key_length = max(len(str(key)) for key in stats.keys())
    
    for key, value in stats.items():
        print(f"{key:<{max_key_length + 2}} : {value}")
    
    print("="*60)


def get_directory_structure(base_path: Path, max_depth: int = 3) -> str:
    """Get a formatted directory structure."""
    def format_tree(path: Path, prefix: str = "", depth: int = 0) -> List[str]:
        if depth > max_depth:
            return []
        
        lines = []
        try:
            items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
            
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                next_prefix = "    " if is_last else "│   "
                
                if item.is_dir():
                    lines.append(f"{prefix}{current_prefix}{item.name}/")
                    lines.extend(format_tree(item, prefix + next_prefix, depth + 1))
                else:
                    lines.append(f"{prefix}{current_prefix}{item.name}")
                    
        except PermissionError:
            lines.append(f"{prefix}└── [Permission denied]")
        except Exception as e:
            lines.append(f"{prefix}└── [Error: {e}]")
        
        return lines
    
    try:
        lines = [f"{base_path.name}/"]
        lines.extend(format_tree(base_path))
        return "\n".join(lines)
    except Exception as e:
        return f"Error generating structure: {e}"


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Show project status and statistics")
    parser.add_argument("--structure", action="store_true", help="Show directory structure")
    parser.add_argument("--recent", type=int, default=7, help="Days to look back for recent activity")
    parser.add_argument("--depth", type=int, default=3, help="Depth for directory structure")
    
    args = parser.parse_args()
    
    base_path = Path.cwd()
    
    print(f"Data Validation Generator - Project Status")
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get all statistics
    directory_stats = get_directory_stats(base_path)
    file_stats = get_file_stats(base_path)
    recent_activity = get_recent_activity(base_path, args.recent)
    config_status = get_config_status(base_path)
    
    # Print statistics
    print_status_table(directory_stats, "Directory Statistics")
    print_status_table(file_stats, "File Statistics")
    print_status_table(recent_activity, f"Recent Activity (Last {args.recent} Days)")
    print_status_table(config_status, "Configuration Status")
    
    # Show directory structure if requested
    if args.structure:
        print("\n" + "="*60)
        print("DIRECTORY STRUCTURE")
        print("="*60)
        structure = get_directory_structure(base_path, args.depth)
        print(structure)
    
    print(f"\n✅ Project status report completed")


if __name__ == "__main__":
    main() 