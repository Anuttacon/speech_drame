#!/usr/bin/env python3
"""
Checkpoint management utilities for evaluation scripts.
"""

import argparse
import json
import os
import glob
from datetime import datetime
from pathlib import Path
from typing import Optional

def list_checkpoints(checkpoint_dir: str, model_name: Optional[str] = None, prompt_type: Optional[str] = None):
    """List available checkpoints with details"""
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist")
        return
    
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.json"))
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoint(s) in {checkpoint_dir}:")
    print("-" * 80)
    
    for checkpoint_file in sorted(checkpoint_files, key=os.path.getmtime, reverse=True):
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            
            # Filter by model_name and prompt_type if specified
            if model_name and data['args']['model_name'] != model_name:
                continue
            if prompt_type and data['args']['prompt_type'] != prompt_type:
                continue
            
            filename = os.path.basename(checkpoint_file)
            timestamp = datetime.fromtimestamp(data['timestamp'])
            total_processed = data['total_processed']
            model_name_checkpoint = data['args']['model_name']
            prompt_type_checkpoint = data['args']['prompt_type']
            prompt_version = data['args']['prompt_version']
            
            print(f"File: {filename}")
            print(f"  Model: {model_name_checkpoint}")
            print(f"  Prompt Type: {prompt_type_checkpoint} ({prompt_version})")
            print(f"  Processed: {total_processed} items")
            print(f"  Timestamp: {timestamp}")
            print(f"  Size: {os.path.getsize(checkpoint_file) / 1024:.1f} KB")
            print()
            
        except Exception as e:
            print(f"Error reading {checkpoint_file}: {e}")
            print()

def show_checkpoint_details(checkpoint_path: str):
    """Show detailed information about a specific checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file {checkpoint_path} does not exist")
        return
    
    try:
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        
        print(f"Checkpoint Details: {checkpoint_path}")
        print("=" * 60)
        print(f"Model: {data['args']['model_name']}")
        print(f"Model Tag: {data['args']['model_tag']}")
        print(f"Prompt Type: {data['args']['prompt_type']}")
        print(f"Prompt Version: {data['args']['prompt_version']}")
        print(f"Batch Size: {data['args']['batch_size']}")
        print(f"Use Schema: {data['args']['use_schema']}")
        print(f"Total Processed: {data['total_processed']}")
        print(f"Timestamp: {datetime.fromtimestamp(data['timestamp'])}")
        print(f"Start Time: {datetime.fromtimestamp(data['start_time'])}")
        print(f"Duration: {data['timestamp'] - data['start_time']:.2f} seconds")
        
        # Show sample results
        if data['results']:
            print(f"\nSample Results ({min(3, len(data['results']))} of {len(data['results'])}):")
            for i, result in enumerate(data['results'][:3]):
                print(f"  {i+1}. {result.get('id', 'N/A')}: {result.get('response', 'N/A')}")
        
        print(f"\nFile size: {os.path.getsize(checkpoint_path) / 1024:.1f} KB")
        
    except Exception as e:
        print(f"Error reading checkpoint: {e}")

def cleanup_old_checkpoints(checkpoint_dir: str, days: int = 7, dry_run: bool = True):
    """Clean up checkpoints older than specified days"""
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist")
        return
    
    import time
    cutoff_time = time.time() - (days * 24 * 3600)
    
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.json"))
    old_checkpoints = []
    
    for checkpoint_file in checkpoint_files:
        if os.path.getmtime(checkpoint_file) < cutoff_time:
            old_checkpoints.append(checkpoint_file)
    
    if not old_checkpoints:
        print(f"No checkpoints older than {days} days found")
        return
    
    print(f"Found {len(old_checkpoints)} checkpoint(s) older than {days} days:")
    for checkpoint in old_checkpoints:
        timestamp = datetime.fromtimestamp(os.path.getmtime(checkpoint))
        print(f"  {os.path.basename(checkpoint)} ({timestamp})")
    
    if dry_run:
        print(f"\nDry run mode - no files will be deleted")
        print(f"Run with --dry_run False to actually delete these files")
    else:
        print(f"\nDeleting {len(old_checkpoints)} checkpoint(s)...")
        for checkpoint in old_checkpoints:
            try:
                os.remove(checkpoint)
                print(f"  Deleted: {os.path.basename(checkpoint)}")
            except Exception as e:
                print(f"  Error deleting {checkpoint}: {e}")

def get_latest_checkpoint(checkpoint_dir: str, model_name: Optional[str] = None, prompt_type: Optional[str] = None):
    """Get the path to the latest checkpoint matching criteria"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.json"))
    
    if not checkpoint_files:
        return None
    
    # Filter and sort by modification time
    valid_checkpoints = []
    for checkpoint_file in checkpoint_files:
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            
            # Filter by criteria
            if model_name and data['args']['model_name'] != model_name:
                continue
            if prompt_type and data['args']['prompt_type'] != prompt_type:
                continue
            
            valid_checkpoints.append(checkpoint_file)
        except:
            continue
    
    if not valid_checkpoints:
        return None
    
    # Return the most recent one
    return max(valid_checkpoints, key=os.path.getmtime)

def main():
    parser = argparse.ArgumentParser(description="Checkpoint management utilities")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available checkpoints')
    list_parser.add_argument('checkpoint_dir', help='Checkpoint directory')
    list_parser.add_argument('--model_name', help='Filter by model name')
    list_parser.add_argument('--prompt_type', help='Filter by prompt type')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show checkpoint details')
    show_parser.add_argument('checkpoint_path', help='Path to checkpoint file')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old checkpoints')
    cleanup_parser.add_argument('checkpoint_dir', help='Checkpoint directory')
    cleanup_parser.add_argument('--days', type=int, default=7, help='Delete checkpoints older than N days')
    cleanup_parser.add_argument('--dry_run', type=bool, default=True, help='Dry run mode (default: True)')
    
    # Latest command
    latest_parser = subparsers.add_parser('latest', help='Get latest checkpoint path')
    latest_parser.add_argument('checkpoint_dir', help='Checkpoint directory')
    latest_parser.add_argument('--model_name', help='Filter by model name')
    latest_parser.add_argument('--prompt_type', help='Filter by prompt type')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_checkpoints(args.checkpoint_dir, args.model_name, args.prompt_type)
    elif args.command == 'show':
        show_checkpoint_details(args.checkpoint_path)
    elif args.command == 'cleanup':
        cleanup_old_checkpoints(args.checkpoint_dir, args.days, args.dry_run)
    elif args.command == 'latest':
        latest = get_latest_checkpoint(args.checkpoint_dir, args.model_name, args.prompt_type)
        if latest:
            print(latest)
        else:
            print("No matching checkpoint found")
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 