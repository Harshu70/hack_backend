// src/SalesDashboard.jsx

import React from 'react';
import SalesForecast from './SalesForcast';
import TopProducts from './TopProducts';
import SalesKPIs from './SalesKpis';
import DemandForecast from './DemandForecast';
import MonthlySalesChart from './MonthlySales';

function SalesDashboard() {
  return (
    <div className="space-y-6">
      
      {/* Sales KPIs at the top */}
      <SalesKPIs />

      {/* Main sales forecast chart */}
      <SalesForecast />

      {/* Bottom section with Top Products and Demand Forecast side-by-side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <TopProducts />
        <DemandForecast />
      </div>
      <div>
        <MonthlySalesChart/>
      </div>

    </div>
  );
}

export default SalesDashboard;