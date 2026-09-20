import { Excalidraw } from '@excalidraw/excalidraw'
import '@excalidraw/excalidraw/index.css'
import './App.css'
import { studyInitialData } from './studyScene'

function App() {
  return (
    <div className="excalidraw-wrapper">
      <Excalidraw
        initialData={studyInitialData}
        name="Regresion lineal - apuntes"
      />
    </div>
  )
}

export default App
