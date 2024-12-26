import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.jsx";
import "@fontsource/space-grotesk/400.css"; // Regular weight
import "@fontsource/space-grotesk/700.css"; // Bold weight

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
