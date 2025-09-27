import React from 'react';
import ChurnPrediction from './ChurnPrediction';
import ChurnTrendChart from './ChurnTrendChart';
import ChurnSegment from './ChurnSegment';
import SalesForecast from './SalesForcast';
import TopProducts from './TopProducts';
import SalesKPIs from './SalesKpis';
import DemandForecast from './DemandForecast';
import GeoChart from './GeoChart';
import DataUploader from './DataUploader';

function App() {
  return (
    <div className="bg-gray-100 min-h-screen">
      <div className="container mx-auto p-4 sm:p-6 lg:p-8">
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-gray-800">Customer Churn Dashboard</h1>
        </header>

        <div className=" gap-8">
            <ChurnPrediction />
          <div className="lg:col-span-3">
            <ChurnTrendChart />
          </div>

          {/* Side components take up 2/5 */}
          <div className="lg:col-span-2 space-y-8">
            <ChurnSegment />
          </div>
        </div>
            <SalesForecast />
            <TopProducts />
            <SalesKPIs />
            <DemandForecast />
            <GeoChart />
            <DataUploader />
      </div>
    </div>
  );
}

export default App;