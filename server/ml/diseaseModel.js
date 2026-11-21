const fs = require('fs');
const axios = require('axios');
const FormData = require('form-data');

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || process.env.DISEASE_MODEL_URL || 'http://127.0.0.1:8001';

/**
 * Classify a plant disease image via the FastAPI ML microservice.
 *
 * @param {Buffer|string} imageBuffer - Image file buffer or path to image on disk
 * @param {string} [filename='leaf.jpg'] - Filename for form-data upload
 * @returns {Promise<Object>}
 */
async function classifyDisease(imageBuffer, filename = 'leaf.jpg', crop = null, species = null) {
  let buf = imageBuffer;
  if (typeof imageBuffer === 'string') {
    buf = fs.readFileSync(imageBuffer);
  }

  const form = new FormData();
  form.append('file', buf, filename);
  if (crop) form.append('crop', crop);
  if (species) form.append('species', species);

  const response = await axios.post(`${ML_SERVICE_URL}/predict`, form, {
    headers: form.getHeaders(),
    timeout: 15000,
  });
  return response.data;
}

/**
 * Stage 1 inference wrapper for diseaseController.
 * Returns null gracefully if microservice is offline or in test mode.
 */
async function predict(imagePath, crop = null, species = null) {
  if (process.env.NODE_ENV === 'test') return null;
  try {
    const data = await classifyDisease(imagePath, 'leaf.jpg', crop, species);
    if (!data) return null;
    return {
      status: data.status,
      className: data.class_name,
      confidence: typeof data.confidence === 'number' ? Math.round(data.confidence * 10000) / 100 : null,
      top_3: data.top_3 || [],
      rawClassName: data.raw_class_name,
      rawConfidence: typeof data.raw_confidence === 'number' ? Math.round(data.raw_confidence * 10000) / 100 : null,
      is_supported_plant: data.is_supported_plant ?? true,
      crop_mismatch: Boolean(data.crop_mismatch),
      selected_crop: data.selected_crop || crop,
      detected_crop: data.detected_crop || null,
      detected_crop_si: data.detected_crop_si || null,
      identified_plant: data.identified_plant || null,
      ood_score: data.ood_score,
      plant_check: data.plant_check,
      message_en: data.message_en,
      message_si: data.message_si,
    };
  } catch (err) {
    console.warn('[disease-ml] ML microservice not reachable or error:', err.message);
    return null;
  }
}

module.exports = { classifyDisease, predict };
