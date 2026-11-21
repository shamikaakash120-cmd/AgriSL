// diseaseModelClient — Node.js client adapter for the custom-trained disease
// classification microservice (Python FastAPI, ml-service/main.py).

const { classifyDisease: callMicroservice } = require('../ml/diseaseModel');

function enabled() {
  const url = process.env.ML_SERVICE_URL || process.env.DISEASE_MODEL_URL;
  return Boolean(url) && process.env.NODE_ENV !== 'test';
}

/**
 * Send the image to the Python FastAPI /predict endpoint and return normalized result.
 *
 * @param {string} imagePath  — absolute path on disk (from multer)
 * @param {string} [mimetype] — e.g. "image/jpeg"
 * @param {string} [cropType] — farmer-selected crop
 * @param {string} [species]  — optional botanical species string
 * @returns {Promise<Object|null>}
 */
async function classifyDisease(imagePath, mimetype, cropType, species = null) {
  if (!enabled()) return null;

  try {
    const d = await callMicroservice(imagePath, 'leaf.jpg', cropType, species);
    if (!d) return null;

    if (d.status === 'unsupported_crop' || d.is_supported_plant === false) {
      return {
        status: 'unsupported_crop',
        is_supported_plant: false,
        crop: null,
        disease: null,
        isHealthy: false,
        probability: null,
        confidence: null,
        classIndex: null,
        class_name: null,
        className: null,
        raw_class_name: d.raw_class_name || null,
        raw_confidence: typeof d.raw_confidence === 'number' ? d.raw_confidence : null,
        crop_mismatch: false,
        selected_crop: d.selected_crop || cropType,
        detected_crop: null,
        detected_crop_si: null,
        identified_plant: d.identified_plant || null,
        ood_score: typeof d.ood_score === 'number' ? d.ood_score : null,
        plant_check: d.plant_check || 'unavailable',
        message_en: d.message_en || null,
        message_si: d.message_si || null,
        top_3: [],
      };
    }

    // Normal supported / crop_mismatch response
    const CROPS = ['Bell_Pepper', 'Banana', 'Corn', 'Potato', 'Rice', 'Tea', 'Tomato'];
    const prefix = CROPS.find((p) => d.class_name?.startsWith(p + '_'));
    const crop = prefix ? prefix.replace('_', ' ') : d.class_name?.split('_')[0];
    const diseaseName = prefix
      ? d.class_name.slice(prefix.length + 1).replace(/_/g, ' ')
      : d.class_name?.replace(/_/g, ' ');
    const isHealthy = d.class_name?.toLowerCase().includes('healthy');

    return {
      status: d.status || (d.crop_mismatch ? 'crop_mismatch' : 'ok'),
      crop,
      disease: isHealthy ? 'Healthy' : diseaseName,
      isHealthy,
      probability: typeof d.confidence === 'number' ? d.confidence : null,
      confidence: typeof d.confidence === 'number' ? Math.round(d.confidence * 10000) / 100 : null,
      classIndex: null,
      class_name: d.class_name,
      className: d.class_name,
      raw_class_name: d.raw_class_name,
      raw_confidence: d.raw_confidence,
      is_supported_plant: d.is_supported_plant ?? true,
      crop_mismatch: Boolean(d.crop_mismatch),
      selected_crop: d.selected_crop || cropType,
      detected_crop: d.detected_crop || null,
      detected_crop_si: d.detected_crop_si || null,
      identified_plant: d.identified_plant || null,
      ood_score: typeof d.ood_score === 'number' ? d.ood_score : null,
      plant_check: d.plant_check || 'unavailable',
      message_en: d.message_en || null,
      message_si: d.message_si || null,
      top_3: d.top_3 || [],
    };
  } catch (err) {
    console.warn('[diseaseModelClient] Error calling ML microservice:', err.message);
    return null;
  }
}

module.exports = { classifyDisease, enabled };
