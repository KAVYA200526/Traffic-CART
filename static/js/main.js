// Chart.js configurations and utilities for VANET ML Analysis

// Chart color schemes
const chartColors = {
    primary: '#0d6efd',
    success: '#198754',
    warning: '#ffc107',
    danger: '#dc3545',
    info: '#0dcaf0',
    light: '#f8f9fa',
    dark: '#212529'
};

// Default chart options
const defaultChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            position: 'top',
            labels: {
                usePointStyle: true,
                padding: 20,
                font: {
                    family: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                    size: 12
                }
            }
        },
        tooltip: {
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            titleColor: '#ffffff',
            bodyColor: '#ffffff',
            borderColor: '#ffffff',
            borderWidth: 1,
            cornerRadius: 6,
            displayColors: true
        }
    },
    scales: {
        x: {
            grid: {
                color: 'rgba(0, 0, 0, 0.1)',
                drawBorder: false
            },
            ticks: {
                font: {
                    family: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                    size: 11
                }
            }
        },
        y: {
            grid: {
                color: 'rgba(0, 0, 0, 0.1)',
                drawBorder: false
            },
            ticks: {
                font: {
                    family: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                    size: 11
                }
            }
        }
    }
};

// Create performance comparison chart
function createPerformanceChart(canvasId, data, type = 'bar') {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    const chart = new Chart(ctx, {
        type: type,
        data: data,
        options: {
            ...defaultChartOptions,
            plugins: {
                ...defaultChartOptions.plugins,
                title: {
                    display: true,
                    text: 'Model Performance Comparison',
                    font: {
                        size: 16,
                        weight: 'bold'
                    },
                    padding: 20
                }
            }
        }
    });
    
    return chart;
}

// Create EDA visualization charts
function createEDACharts() {
    // Priority Level Distribution
    const priorityCtx = document.getElementById('priorityChart');
    if (priorityCtx) {
        new Chart(priorityCtx, {
            type: 'doughnut',
            data: {
                labels: ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'],
                datasets: [{
                    data: [20, 25, 30, 15, 10], // Example data
                    backgroundColor: [
                        chartColors.danger,
                        chartColors.warning,
                        chartColors.info,
                        chartColors.primary,
                        chartColors.success
                    ],
                    borderWidth: 2,
                    borderColor: '#ffffff'
                }]
            },
            options: {
                ...defaultChartOptions,
                plugins: {
                    ...defaultChartOptions.plugins,
                    title: {
                        display: true,
                        text: 'Priority Level Distribution',
                        font: { size: 14, weight: 'bold' }
                    }
                }
            }
        });
    }
    
    // Traffic Density Distribution
    const densityCtx = document.getElementById('densityChart');
    if (densityCtx) {
        new Chart(densityCtx, {
            type: 'line',
            data: {
                labels: ['0-20', '21-40', '41-60', '61-80', '81-100'],
                datasets: [{
                    label: 'Vehicle Count',
                    data: [45, 78, 92, 65, 23], // Example data
                    borderColor: chartColors.primary,
                    backgroundColor: chartColors.primary + '20',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                ...defaultChartOptions,
                plugins: {
                    ...defaultChartOptions.plugins,
                    title: {
                        display: true,
                        text: 'Traffic Density Distribution (vehicles/km²)',
                        font: { size: 14, weight: 'bold' }
                    }
                }
            }
        });
    }
}

// Create classification metrics chart
function createClassificationChart(data) {
    const ctx = document.getElementById('classificationChart');
    if (!ctx) return null;
    
    return new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            datasets: [
                {
                    label: 'SVM',
                    data: data.svm || [85, 82, 88, 85],
                    borderColor: chartColors.primary,
                    backgroundColor: chartColors.primary + '20',
                    borderWidth: 2
                },
                {
                    label: 'Random Forest',
                    data: data.rf || [88, 85, 87, 86],
                    borderColor: chartColors.success,
                    backgroundColor: chartColors.success + '20',
                    borderWidth: 2
                },
                {
                    label: 'Hybrid MLP-KNN',
                    data: data.hybrid || [91, 89, 92, 90],
                    borderColor: chartColors.warning,
                    backgroundColor: chartColors.warning + '20',
                    borderWidth: 2
                }
            ]
        },
        options: {
            ...defaultChartOptions,
            plugins: {
                ...defaultChartOptions.plugins,
                title: {
                    display: true,
                    text: 'Classification Performance Radar Chart',
                    font: { size: 16, weight: 'bold' }
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        stepSize: 20,
                        font: { size: 10 }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                }
            }
        }
    });
}

// Create regression metrics chart
function createRegressionChart(data) {
    const ctx = document.getElementById('regressionChart');
    if (!ctx) return null;
    
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['SVM', 'Random Forest', 'Hybrid MLP-KNN'],
            datasets: [
                {
                    label: 'MAE',
                    data: data.mae || [12.5, 10.8, 9.2],
                    backgroundColor: chartColors.danger + '80',
                    borderColor: chartColors.danger,
                    borderWidth: 1
                },
                {
                    label: 'RMSE',
                    data: data.rmse || [18.3, 15.7, 13.9],
                    backgroundColor: chartColors.warning + '80',
                    borderColor: chartColors.warning,
                    borderWidth: 1
                },
                {
                    label: 'R² Score × 100',
                    data: data.r2 || [78.5, 82.3, 85.7],
                    backgroundColor: chartColors.success + '80',
                    borderColor: chartColors.success,
                    borderWidth: 1
                }
            ]
        },
        options: {
            ...defaultChartOptions,
            plugins: {
                ...defaultChartOptions.plugins,
                title: {
                    display: true,
                    text: 'Regression Performance Comparison',
                    font: { size: 16, weight: 'bold' }
                }
            }
        }
    });
}

// Utility function to format numbers for charts
function formatNumber(num, decimals = 2) {
    return parseFloat(num).toFixed(decimals);
}

// Utility function to create gradient backgrounds
function createGradient(ctx, color1, color2) {
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, color1);
    gradient.addColorStop(1, color2);
    return gradient;
}

// Animation configurations
const animationConfig = {
    duration: 2000,
    easing: 'easeOutQuart'
};

// Initialize charts when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize EDA charts if on EDA page
    if (document.getElementById('priorityChart')) {
        createEDACharts();
    }
    
    // Auto-resize charts on window resize
    window.addEventListener('resize', function() {
        Chart.helpers.each(Chart.instances, function(instance) {
            instance.resize();
        });
    });
});

// Export functions for use in other scripts
window.chartUtils = {
    createPerformanceChart,
    createEDACharts,
    createClassificationChart,
    createRegressionChart,
    formatNumber,
    createGradient,
    chartColors,
    defaultChartOptions,
    animationConfig
};
