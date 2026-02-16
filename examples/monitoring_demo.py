"""
Demonstration of the MonitoringDashboard for real-time metrics tracking.

This example shows how to use the MonitoringDashboard to track and display
metrics during encryption/decryption operations.
"""

import time
from fourier_encryption.visualization.monitoring_dashboard import MonitoringDashboard
from fourier_encryption.models.data_models import Metrics


def simulate_encryption_process():
    """
    Simulate an encryption process with progress updates.
    
    Demonstrates how to use the MonitoringDashboard to track metrics
    during a long-running operation.
    """
    print("Starting encryption simulation with monitoring dashboard...\n")
    
    # Create dashboard with 0.1 second update interval
    dashboard = MonitoringDashboard(update_interval=0.1)
    
    # Start background monitoring
    dashboard.start_monitoring()
    
    try:
        # Simulate processing 100 coefficients
        total_coefficients = 100
        
        for i in range(total_coefficients + 1):
            # Update metrics for current processing state
            metrics = Metrics(
                current_coefficient_index=i,
                active_radius=10.0 * (1.0 - i / total_coefficients),  # Decreasing radius
                progress_percentage=(i / total_coefficients) * 100.0,
                fps=30.0,
                processing_time=time.time() - dashboard._start_time,
                memory_usage_mb=dashboard.metrics.memory_usage_mb,  # Use system-collected value
                encryption_status="encrypting" if i < total_coefficients else "complete"
            )
            
            dashboard.update_metrics(metrics)
            
            # Display dashboard every 10 coefficients
            if i % 10 == 0:
                print("\n" + dashboard.display(mode="terminal"))
            
            # Simulate processing time
            time.sleep(0.05)
        
        # Final display
        print("\n" + dashboard.display(mode="terminal"))
        print("\nEncryption simulation complete!")
        
    finally:
        # Stop monitoring
        dashboard.stop_monitoring()


def demonstrate_context_manager():
    """
    Demonstrate using the dashboard as a context manager.
    
    The context manager automatically starts and stops monitoring.
    """
    print("\n\nDemonstrating context manager usage...\n")
    
    with MonitoringDashboard(update_interval=0.1) as dashboard:
        # Simulate quick operation
        for i in range(5):
            metrics = Metrics(
                current_coefficient_index=i * 20,
                active_radius=5.0,
                progress_percentage=i * 20.0,
                fps=30.0,
                processing_time=i * 0.5,
                memory_usage_mb=100.0,
                encryption_status="encrypting"
            )
            dashboard.update_metrics(metrics)
            time.sleep(0.2)
        
        print(dashboard.display(mode="terminal"))
    
    print("\nContext manager automatically stopped monitoring.")


def demonstrate_gui_mode():
    """
    Demonstrate GUI mode output (JSON format).
    
    This format is useful for integration with web dashboards or GUI applications.
    """
    print("\n\nDemonstrating GUI mode (JSON output)...\n")
    
    dashboard = MonitoringDashboard()
    
    metrics = Metrics(
        current_coefficient_index=50,
        active_radius=7.5,
        progress_percentage=50.0,
        fps=30.0,
        processing_time=5.0,
        memory_usage_mb=150.0,
        encryption_status="encrypting"
    )
    
    dashboard.update_metrics(metrics)
    
    print("JSON output for GUI integration:")
    print(dashboard.display(mode="gui"))


def demonstrate_custom_callback():
    """
    Demonstrate using a custom update callback.
    
    The callback is invoked periodically by the monitoring thread
    to fetch updated metrics automatically.
    """
    print("\n\nDemonstrating custom update callback...\n")
    
    dashboard = MonitoringDashboard(update_interval=0.2)
    
    # Shared state for callback
    state = {"index": 0}
    
    def get_current_metrics():
        """Custom callback that returns updated metrics."""
        state["index"] += 1
        return Metrics(
            current_coefficient_index=state["index"],
            active_radius=10.0 - state["index"] * 0.1,
            progress_percentage=min(state["index"] * 2.0, 100.0),
            fps=30.0,
            processing_time=state["index"] * 0.2,
            memory_usage_mb=100.0 + state["index"],
            encryption_status="encrypting" if state["index"] < 50 else "complete"
        )
    
    # Set callback and start monitoring
    dashboard.set_update_callback(get_current_metrics)
    dashboard.start_monitoring()
    
    try:
        # Let the callback update metrics automatically
        for _ in range(3):
            time.sleep(1.0)
            print(dashboard.display(mode="terminal"))
    finally:
        dashboard.stop_monitoring()
    
    print("\nCallback-based monitoring complete!")


if __name__ == "__main__":
    # Run all demonstrations
    simulate_encryption_process()
    demonstrate_context_manager()
    demonstrate_gui_mode()
    demonstrate_custom_callback()
    
    print("\n" + "=" * 60)
    print("All monitoring dashboard demonstrations complete!")
    print("=" * 60)
