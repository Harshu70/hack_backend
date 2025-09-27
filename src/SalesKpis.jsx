import React, { useState, useEffect } from 'react';

function SalesKPIs() {
  const [kpis, setKpis] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('http://127.0.0.1:5000/api/sales_kpis')
      .then(res => res.json())
      .then(data => {
        setKpis(data);
        setLoading(false);
      })
      .catch(error => {
        console.error("Error fetching sales KPIs:", error);
        setLoading(false);
      });
  }, []);

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 0,
    }).format(value);
  };

  if (loading) {
    return <div className="text-center p-4">Loading KPIs...</div>;
  }
  if (!kpis) return null;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h3 className="text-gray-500 text-sm font-medium">Total Revenue</h3>
        <p className="text-3xl font-bold text-gray-800">{formatCurrency(kpis.total_revenue)}</p>
      </div>
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h3 className="text-gray-500 text-sm font-medium">Best Month ({kpis.best_month})</h3>
        <p className="text-3xl font-bold text-green-600">{formatCurrency(kpis.best_month_sales)}</p>
      </div>
       <div className="bg-white p-6 rounded-lg shadow-md">
        <h3 className="text-gray-500 text-sm font-medium">Avg. Daily Sales</h3>
        <p className="text-3xl font-bold text-gray-800">{formatCurrency(kpis.average_daily_sales)}</p>
      </div>
    </div>
  );
}

export default SalesKPIs;