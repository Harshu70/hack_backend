// import React from 'react';
// import ChurnPrediction from './ChurnPrediction';
// import ChurnTrendChart from './ChurnTrendChart';
// import ChurnSegment from './ChurnSegment';
// import SalesForecast from './SalesForcast';
// import TopProducts from './TopProducts';
// import SalesKPIs from './SalesKpis';
// import DemandForecast from './DemandForecast';
// import GeoChart from './GeoChart';
// import DataUploader from './DataUploader';
// import AgeChart from './AgeChart';

    // <div className="bg-gray-100 min-h-screen">
    //   <div className="container mx-auto p-4 sm:p-6 lg:p-8">
    //     <header className="mb-8">
    //       <h1 className="text-3xl font-bold text-gray-800">Customer Churn Dashboard</h1>
    //     </header>

    //     <div className=" gap-8">
    //         <ChurnPrediction />
    //       <div className="lg:col-span-3">
    //         <ChurnTrendChart />
    //       </div>

    //       {/* Side components take up 2/5 */}
    //       <div className="lg:col-span-2 space-y-8">
    //         <ChurnSegment />
    //       </div>
    //     </div>
    //         <SalesForecast />
    //         <TopProducts />
    //         <SalesKPIs />
    //         <DemandForecast />
    //         <GeoChart />
    //         <DataUploader />
    //         <AgeChart />
    //   </div>
    // </div>
  
import { useState } from 'react';
import ChrunDashboard from './ChrunDashboard.jsx';
import SalesDashboard from './SalesDashboard.jsx';
import FileUpload from './FileUpload.jsx';

function App() {
  // State to manage which tab is currently active
  const [activeTab, setActiveTab] = useState('churn');

  // Function to render the correct component based on the active tab
  const renderContent = () => {
    switch (activeTab) {
      case 'churn':
        return <ChrunDashboard />;
      case 'sales':
        return <SalesDashboard />;
      case 'upload':
        return <FileUpload />;
      default:
        return <ChrunDashboard />;
    }
  };

  // Helper function for button styling
  const getButtonClass = (tabName) => {
    return activeTab === tabName
      ? 'px-4 py-2 font-semibold text-white bg-blue-600 rounded-md shadow-md'
      : 'px-4 py-2 font-semibold text-gray-700 bg-gray-200 hover:bg-gray-300 rounded-md';
  };

  return (
    <div className="bg-gray-100 min-h-screen">
      <div className="container mx-auto p-4 sm:p-6 lg:p-8">
        
        {/* Header and Navigation */}
        <header className="mb-8 p-4 bg-white rounded-lg shadow-md">
          <div className="flex flex-col sm:flex-row justify-between items-center">
            <h1 className="text-3xl font-bold text-gray-800 mb-4 sm:mb-0">
              Analytics Dashboard
            </h1>
            <nav className="flex space-x-2 sm:space-x-4">
              <button onClick={() => setActiveTab('churn')} className={getButtonClass('churn')}>
                Customer Churn
              </button>
              <button onClick={() => setActiveTab('sales')} className={getButtonClass('sales')}>
                Sales Forecasting
              </button>
              <button onClick={() => setActiveTab('upload')} className={getButtonClass('upload')}>
                Upload Data
              </button>
            </nav>
          </div>
        </header>

        {/* Main content area where components are rendered */}
        <main>
          {renderContent()}
        </main>
        
      </div>
    </div>
  );
}

export default App;