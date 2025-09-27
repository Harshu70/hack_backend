import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  // 'useState' hook to store the orders data
  const [orders, setOrders] = useState([]);
  // 'useState' hook to handle loading state
  const [loading, setLoading] = useState(true);

  // 'useEffect' hook to fetch data when the component loads
  useEffect(() => {
    // The URL of your Flask API endpoint
    fetch('http://127.0.0.1:5000/api/orders')
      .then(response => response.json())
      .then(data => {
        setOrders(data); // Update the state with the fetched data
        setLoading(false); // Set loading to false
      })
      .catch(error => {
        console.error("Error fetching data:", error);
        setLoading(false); // Also set loading to false on error
      });
  }, []); // The empty array means this effect runs only once

  // Display a loading message while data is being fetched
  if (loading) {
    return <div>Loading orders...</div>;
  }

  // Render the fetched data in a simple table
  return (
    <div className="App">
      <h1>Customer Orders</h1>
      <table>
        <thead>
          <tr>
            <th>Order ID</th>
            <th>Customer ID</th>
            <th>Product ID</th>
            <th>Last Purchase Date</th>
            <th>Price</th>
            <th>Quantity</th>
          </tr>
        </thead>
        <tbody>
          {orders.map(order => (
            <tr key={order.order_id}>
              <td>{order.order_id}</td>
              <td>{order.customer_id}</td>
              <td>{order.product_id}</td>
              <td>{order.last_purchase_date}</td>
              <td>{order.unit_price}</td>
              <td>{order.quantity}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default App;