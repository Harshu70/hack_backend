import React, { useState, useEffect } from 'react';

function ChurnPrediction() {
  const [churners, setChurners] = useState([]);
  const [loading, setLoading] = useState(true);
  const [count, setCount] = useState(10); 

  useEffect(() => {
    setLoading(true);
    fetch(`http://127.0.0.1:5000/api/predict_churn?count=${count}`)
      .then(response => response.json())
      .then(data => {
        setChurners(data);
        setLoading(false);
      })
      .catch(error => {
        console.error("Error fetching churn predictions:", error);
        setLoading(false);
      });
  }, [count]);

  return (
    // We make the main container a flex column with a fixed height
    <div className="bg-white p-4 md:p-6 rounded-lg shadow-md flex flex-col h-[600px]">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold text-gray-800">High-Risk Customers</h2>
        <select 
          value={count} 
          onChange={(e) => setCount(Number(e.target.value))}
          className="border border-gray-300 rounded-md p-1"
        >
          <option value="5">Top 5</option>
          <option value="10">Top 10</option>
          <option value="20">Top 20</option>
          <option value="50">Top 50</option>
        </select>
      </div>

      {/* This container will handle the scrolling */}
      {/* We replace 'flex-grow' with 'overflow-y-auto' which will add the scrollbar */}
      <div className="overflow-y-auto">
        {loading ? (
          <div className="text-center p-4">Loading...</div>
        ) : (
          <table className="w-full text-left">
            <thead className="bg-gray-50 sticky top-0">
              <tr>
                <th className="p-3 text-sm font-semibold text-gray-600 uppercase tracking-wider">Customer ID</th>
                <th className="p-3 text-sm font-semibold text-gray-600 uppercase tracking-wider">Churn Probability</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {churners.map(customer => (
                <tr key={customer.customer_id} className="hover:bg-gray-50">
                  <td className="p-3 whitespace-nowrap">{customer.customer_id}</td>
                  <td className="p-3 whitespace-nowrap font-medium text-red-600">
                    {(customer.churn_probability * 100).toFixed(2)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

export default ChurnPrediction;