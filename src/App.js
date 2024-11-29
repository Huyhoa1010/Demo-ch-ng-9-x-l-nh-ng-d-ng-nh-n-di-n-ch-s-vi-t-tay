import React, { useState, useEffect } from 'react';
import './App.css';
import CanvasGrid from './canvasGrid';
import * as tf from '@tensorflow/tfjs';
import { loadGraphModel } from '@tensorflow/tfjs';

function App() {
  const [canvas, setCanvas] = useState(Array(28).fill(Array(28).fill(0)));
  const [model, setModel] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);

  useEffect(() => {
    async function loadModel() {
      const loadedModel = await loadGraphModel('https://raw.githubusercontent.com/WasinUddy/MNIST-JS/main/tfjs_model/model.json');
      setModel(loadedModel);
    }
    loadModel();
  }, []);

  function preprocessImage(inputImage) {
    const flatArray = inputImage.flatMap(row => row);
    const tensor = tf.tensor4d(flatArray, [1, 28, 28, 1]);
    return tensor;
  }

  async function predictImage() {
    if (!model) {
      console.error('Model not loaded');
      return;
    }

    const inputTensor = preprocessImage(canvas);
    const prediction = model.predict(inputTensor);
    const result = Array.from(prediction.dataSync());

    // Find the index with the maximum value
    const maxIndex = result.reduce((maxIndex, currValue, currIndex, arr) => currValue > arr[maxIndex] ? currIndex : maxIndex, 0);

    setPredictionResult(maxIndex);

    inputTensor.dispose();
  }

  return (
    <div className="App">
      <h1 className="app-title">DỰ ĐOÁN CHỮ SỐ VIẾT TAY</h1>
      <h2 className="subtitle">Vẽ một chữ số bất kỳ và nhấn dự đoán</h2>
      <div className="canvas-container">
        <CanvasGrid canvas={canvas} setCanvas={setCanvas} />
      </div>

      <div className="button-container">
        <button className="clear-button" onClick={() => setCanvas(Array(28).fill(Array(28).fill(0)))}>Xóa</button>
        <button className="predict-button" onClick={() => predictImage()}>Dự đoán</button>
      </div>

      {predictionResult !== null && (
        <div className="prediction-display">
          <p className="prediction-text">Chữ số dự đoán: {predictionResult}</p>
        </div>
      )}
    </div>
  );
}

export default App;
