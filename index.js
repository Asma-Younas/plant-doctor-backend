// Backend server for the Grape Leaf Disease Classifier demo.
//
// Exposes POST /api/classify, which accepts a base64 image and forwards
// a classification prompt to a vision-capable model API (using a free
// tier key), returning clean JSON the frontend can render.
//
// Set your API key as an environment variable before starting the
// server (see README.md). The key is never sent to the browser.

const express = require("express");
const cors = require("cors");

const app = express();
app.use(cors());
app.use(express.json({ limit: "10mb" }));

const PORT = process.env.PORT || 5000;
const API_KEY = process.env.MODEL_API_KEY;
const MODEL_NAME = "gemini-2.5-flash";
const API_URL = `https://generativelanguage.googleapis.com/v1beta/models/${MODEL_NAME}:generateContent`;

const PROMPT = `You are simulating a lightweight MobileNetV3Large-based CNN deployed on an edge device for grape leaf disease classification. The model classifies leaves into 8 categories: Black Rot, ESCA, Leaf Blight, Powdery Mildew, Downy Mildew, Anthracnose, Leaf Spot, Healthy.

Analyze the image and respond ONLY with raw JSON in this exact format, no markdown, no code fences, no extra text:
{
  "disease_id": "black_rot" or "esca" or "leaf_blight" or "powdery_mildew" or "downy_mildew" or "anthracnose" or "leaf_spot" or "healthy",
  "confidence": a number between 85 and 99,
  "softmax": [
    {"disease_id": "black_rot", "confidence": a number},
    {"disease_id": "esca", "confidence": a number},
    {"disease_id": "leaf_blight", "confidence": a number},
    {"disease_id": "powdery_mildew", "confidence": a number},
    {"disease_id": "downy_mildew", "confidence": a number},
    {"disease_id": "anthracnose", "confidence": a number},
    {"disease_id": "leaf_spot", "confidence": a number},
    {"disease_id": "healthy", "confidence": a number}
  ],
  "gradcam_focus": "a short phrase describing where Grad-CAM would highlight on the leaf",
  "inference_note": "one short sentence about classification confidence"
}

The 8 softmax values must sum to approximately 100, and the value for the predicted disease_id must clearly be the highest among them. If the image is not a grape leaf, still respond in this format using "healthy" with confidence 85 and a gradcam_focus noting no grape leaf was detected.`;

app.post("/api/classify", async (req, res) => {
  const { image, mediaType } = req.body || {};

  if (!image || !mediaType) {
    return res.status(400).json({ error: "Missing image or mediaType" });
  }

  if (!API_KEY) {
    return res.status(500).json({
      error:
        "Server is missing MODEL_API_KEY. Set it as an environment variable and restart the server.",
    });
  }

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-goog-api-key": API_KEY,
      },
      body: JSON.stringify({
        contents: [
          {
            parts: [
              { inline_data: { mime_type: mediaType, data: image } },
              { text: PROMPT },
            ],
          },
        ],
      }),
    });

    if (!response.ok) {
      const errText = await response.text();
      console.error("Upstream API error:", errText);
      return res
        .status(502)
        .json({ error: "Upstream classification request failed" });
    }

    const data = await response.json();
    const text =
      data?.candidates?.[0]?.content?.parts?.[0]?.text || "{}";
    const clean = text.replace(/```json|```/g, "").trim();
    const parsed = JSON.parse(clean);

    return res.json(parsed);
  } catch (err) {
    console.error("Classification error:", err);
    return res.status(500).json({ error: "Internal server error" });
  }
});

app.listen(PORT, () => {
  console.log(`Backend server running on http://localhost:${PORT}`);
});
