import React, { useState } from 'react';
import {datos} from '../datos'

function App() {
  const [response, setResponse] = useState(null);

  const sendData = async () => {
    const data = { name: 'Juan', age: 30 };

    try {
      const response = await fetch('http://127.0.0.1:5000/data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(datos)
      });

      const result = await response.json();
      setResponse(result);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div>
      <button onClick={sendData}>Enviar Datos</button>
      {response && <p>Respuesta del servidor: {JSON.stringify(response)}</p>}
    </div>
  );
}

export default App;
