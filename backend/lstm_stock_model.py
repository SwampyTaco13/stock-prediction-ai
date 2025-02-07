import React, { useState, useEffect } from "react";

export default function StockDashboard() {
  const [ticker, setTicker] = useState("AAPL");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  
  const fetchStockPrediction = async () => {
    setLoading(true);
    try {
      const response = await fetch(`http://localhost:5000/predict?ticker=${ticker}`);
      const result = await response.json();
      setData(result);
    } catch (error) {
      console.error("Error fetching prediction:", error);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchStockPrediction();
  }, [ticker]);
  
  return (
    <div className="p-4 max-w-xl mx-auto text-center">
      <h1 className="text-xl font-bold">AI Stock Prediction Dashboard</h1>
      <div className="mt-4">
        <input
          type="text"
          value={ticker}
          onChange={(e) => setTicker(e.target.value.toUpperCase())}
          className="border p-2 rounded w-1/2"
          placeholder="Enter stock ticker (e.g., AAPL)"
        />
        <button
          onClick={fetchStockPrediction}
          className="ml-2 px-4 py-2 bg-blue-500 text-white rounded"
        >
          Fetch Prediction
        </button>
      </div>
      {loading && <p className="mt-4">Loading...</p>}
      {data && (
        <div className="mt-4 border p-4 rounded shadow">
          <p><strong>Ticker:</strong> {data.ticker}</p>
          <p><strong>Predicted Price:</strong> ${data.predicted_price}</p>
          <p><strong>Current Price:</strong> ${data.current_price}</p>
          <p><strong>Recommendation:</strong> {data.recommendation}</p>
        </div>
      )}
    </div>
  );
}
