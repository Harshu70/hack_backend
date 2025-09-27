// src/ChurnDashboard.jsx

import React from 'react';
import ChurnPrediction from './ChurnPrediction';
import ChurnTrendChart from './ChurnTrendChart';
import ChurnSegment from './ChurnSegment';
import GeoChart from './GeoChart';

function ChurnDashboard() {
  return (
    <div className="gap-6">
      
      {/* High-Risk Customers Table - takes up 2/5 of the width on large screens */}
      <div className="lg:col-span-2">
        <ChurnPrediction />
      </div>
      <div>
        <ChurnTrendChart />
      </div>

      {/* Charts Section - takes up 3/5 of the width on large screens */}
      <div className="lg:col-span-3 space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <ChurnSegment />
            <GeoChart />
        </div>
      </div>
      
    </div>
  );
}

export default ChurnDashboard;