#!/usr/bin/env python3
"""
Accuracy vs Speed Trade-off Script

This script demonstrates different Whisper model configurations
for various accuracy vs speed requirements.
"""

from realtime_whisper import RealtimeWhisperRecognizer
import time

def run_high_accuracy():
    """Maximum accuracy configuration (slower but most accurate)"""
    print("🎯 HIGH ACCURACY MODE")
    print("Model: large (1550M parameters)")
    print("Best for: Critical applications, professional use")
    print("Trade-off: Higher CPU usage, slower processing\n")
    
    try:
        with RealtimeWhisperRecognizer(
            model_name="large",           # Best accuracy
            chunk_duration=2.5,           # Longer chunks for better context
            language="ja",
            silence_threshold=0.028,      # Optimized for your environment
            min_speech_duration=1.2       # Longer speech requirement
        ) as recognizer:
            recognizer.start_recording()
            
            while True:
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\n🛑 High accuracy mode stopped")

def run_balanced():
    """Balanced accuracy and speed configuration"""
    print("⚖️ BALANCED MODE")
    print("Model: medium (769M parameters)")
    print("Best for: General use, good balance")
    print("Trade-off: Moderate accuracy, moderate speed\n")
    
    try:
        with RealtimeWhisperRecognizer(
            model_name="medium",          # Good balance
            chunk_duration=2.0,           # Standard chunks
            language="ja",
            silence_threshold=0.028,      # Optimized for your environment
            min_speech_duration=1.0       # Standard speech requirement
        ) as recognizer:
            recognizer.start_recording()
            
            while True:
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\n🛑 Balanced mode stopped")

def run_fast():
    """Fast processing configuration (lower accuracy but faster)"""
    print("⚡ FAST MODE")
    print("Model: tiny (39M parameters)")
    print("Best for: Real-time applications, low CPU usage")
    print("Trade-off: Lower accuracy, faster processing\n")
    
    try:
        with RealtimeWhisperRecognizer(
            model_name="tiny",            # Fastest processing
            chunk_duration=1.5,           # Shorter chunks for responsiveness
            language="ja",
            silence_threshold=0.028,      # Optimized for your environment
            min_speech_duration=0.8       # Shorter speech requirement
        ) as recognizer:
            recognizer.start_recording()
            
            while True:
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\n🛑 Fast mode stopped")

def show_model_comparison():
    """Show comparison of different models"""
    print("🔍 WHISPER MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<8} {'Parameters':<12} {'Speed':<8} {'Accuracy':<10} {'Use Case'}")
    print("-" * 60)
    print(f"{'tiny':<8} {'39M':<12} {'Fastest':<8} {'Low':<10} {'Real-time, low CPU'}")
    print(f"{'base':<8} {'74M':<12} {'Fast':<8} {'Low-Med':<10} {'General use, balanced'}")
    print(f"{'small':<8} {'244M':<12} {'Medium':<8} {'Medium':<10} {'Good accuracy, moderate speed'}")
    print(f"{'medium':<8} {'769M':<12} {'Slow':<8} {'High':<10} {'High accuracy, slower'}")
    print(f"{'large':<8} {'1550M':<12} {'Slowest':<8} {'Highest':<10} {'Maximum accuracy'}")
    print("=" * 60)
    print()

def main():
    """Main menu for model selection"""
    while True:
        print("\n🎤 WHISPER RECOGNITION - MODEL SELECTION")
        print("=" * 50)
        print("1. 🎯 High Accuracy (large model)")
        print("2. ⚖️  Balanced (medium model)")
        print("3. ⚡ Fast (tiny model)")
        print("4. 🔍 Show Model Comparison")
        print("5. 🚪 Exit")
        print("-" * 50)
        
        choice = input("Select mode (1-5): ").strip()
        
        if choice == "1":
            run_high_accuracy()
        elif choice == "2":
            run_balanced()
        elif choice == "3":
            run_fast()
        elif choice == "4":
            show_model_comparison()
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-5.")

if __name__ == "__main__":
    main()
