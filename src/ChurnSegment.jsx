import React, { useState, useEffect } from 'react';
// We'll use the Doughnut chart component
import { Doughnut } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, Title } from 'chart.js';

// Register the necessary components for a doughnut chart
ChartJS.register(ArcElement, Tooltip, Legend, Title);

function ChurnSegment() {
  const [chartData, setChartData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('http://127.0.0.1:5000/api/churn_segmentation')
      .then(response => response.json())
      .then(data => {
        // Extract labels (e.g., "High Risk") and values (e.g., 30)
        const labels = Object.keys(data);
        const values = Object.values(data);

        setChartData({
          labels: labels,
          datasets: [
            {
              label: '# of Customers',
              data: values,
              backgroundColor: [
                'rgba(220, 53, 69, 0.8)',  // Red for High Risk
                'rgba(255, 193, 7, 0.8)',   // Yellow for Medium Risk
                'rgba(25, 135, 84, 0.8)',   // Green for Low Risk
              ],
              borderColor: [
                'rgba(220, 53, 69, 1)',
                'rgba(255, 193, 7, 1)',
                'rgba(25, 135, 84, 1)',
              ],
              borderWidth: 1,
            },
          ],
        });
        setLoading(false);
      })
      .catch(error => {
        console.error("Error fetching segmentation data:", error);
        setLoading(false);
      });
  }, []);

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Customer Segmentation by Churn Risk',
        font: { size: 18 }
      },
    },
  };

  if (loading) {
    return <div className="text-center p-4">Loading segmentation chart...</div>;
  }

  return (
    <div className="bg-white p-4 md:p-6 rounded-lg shadow-md">
      {chartData && <Doughnut options={options} data={chartData} />}
    </div>
  );
}

export default ChurnSegment;