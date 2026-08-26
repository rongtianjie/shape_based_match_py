/**
 * ShapeMatch Studio - Web Frontend Application Logic
 */

// Application State
const state = {
  currentStep: 1,
  currentViewMode: 'template', // 'template' | 'match' | 'split'
  showOverlayLayer: true,

  // Images in memory
  templateFile: null,
  templateImage: null, // HTMLImageElement
  templateOriginalB64: null,
  templateViewB64: null,
  templateDim: { width: 0, height: 0 },

  sourceFile: null,
  sourceImage: null, // HTMLImageElement
  sourceOriginalB64: null,
  matchViewB64: null,
  sourceDim: { width: 0, height: 0 },

  // Parameter defaults (aligned with shape_match.config.py PAT_DEFAULTS and MATCH_DEFAULTS)
  patConfig: {
    contrast_low: 3,
    contrast_high: 5,
    min_contrast: 1,
    min_cont_len: 1,
    num_levels: 1,
    use_polarity: 0,
    angle_start: 0.0,
    angle_extent: 0.0,
    angle_step: 0.0,
  },

  matchConfig: {
    numMatches: 1,
    minScore: 0.15,
    scale_min: 0.8,
    scale_max: 1.2,
    subpixel: 1,
    maxOverLap: 0.5,
    greedness: 0.75,
  },

  // Results telemetry
  featureCount: 0,
  extractDurationMs: 0,
  matches: [],
  selectedMatchIndex: -1,
  hoveredMatchIndex: -1,
  matchDurationMs: 0,

  // Canvas Viewport transform
  zoom: 1.0,
  panX: 0,
  panY: 0,
  isPanning: false,
  startPan: { x: 0, y: 0 },

  // ROI Drag Selection
  isRoiMode: false,
  isSelectingRoi: false,
  roiStart: { x: 0, y: 0 },
  roiEnd: { x: 0, y: 0 },

  // Auto-extraction debounce
  autoExtractTimer: null,
  isExtracting: false,
  isMatching: false,
};

// Parameter default templates (aligned with shape_match/config.py)
const DEFAULT_PAT_CONFIG = {
  contrast_low: 3,
  contrast_high: 5,
  min_contrast: 1,
  min_cont_len: 1,
  num_levels: 1,
  use_polarity: 0,
  angle_start: 0.0,
  angle_extent: 0.0,
  angle_step: 0.0,
};

const DEFAULT_MATCH_CONFIG = {
  subpixel: 1,
  scale_min: 0.8,
  scale_max: 1.2,
  minScore: 0.15,
  maxOverLap: 0.5,
  greedness: 0.75,
  numMatches: 1,
};

// DOM Element Cache
const dom = {};

// The studio can run at / (standalone) or under a mounted sub-application such
// as /shape_match_web. Resolve API calls from the current page path so both
// deployments use the same frontend bundle.
function apiUrl(path) {
  const pagePath = window.location.pathname.endsWith('/')
    ? window.location.pathname
    : `${window.location.pathname}/`;
  return `${pagePath}api/${String(path).replace(/^\/+/, '')}`;
}

async function loadConfigDefaults() {
  try {
    const resp = await fetch(apiUrl('/config/defaults'));
    if (!resp.ok) return;
    const data = await resp.json();
    if (data && data.success) {
      const patDef = data.pat_defaults || data.default_pat;
      const matchDef = data.match_defaults || data.default_match;
      if (patDef) {
        Object.assign(DEFAULT_PAT_CONFIG, patDef);
        if (!state.templateFile && !state.sampleId) {
          state.patConfig = { ...DEFAULT_PAT_CONFIG };
        }
      }
      if (matchDef) {
        Object.assign(DEFAULT_MATCH_CONFIG, matchDef);
        if (!state.sourceFile && !state.sampleId) {
          state.matchConfig = { ...DEFAULT_MATCH_CONFIG };
        }
      }
      updateFormControls();
      validateAllConfigs();
    }
  } catch (err) {
    console.debug('Using built-in config defaults:', err);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  initDomElements();
  initEventListeners();
  initCanvasViewport();
  updateFormControls();
  validateAllConfigs();
  loadConfigDefaults();

  // Default theme is modern light style
  const savedTheme = localStorage.getItem('shapematch_theme') || 'light';
  setTheme(savedTheme);
});

function initDomElements() {
  // Navigation & Header
  dom.stepBtn1 = document.getElementById('step-btn-1');
  dom.stepBtn2 = document.getElementById('step-btn-2');
  dom.step1Panel = document.getElementById('step1-panel');
  dom.step2Panel = document.getElementById('step2-panel');
  dom.btnGotoStep2 = document.getElementById('btn-goto-step2');
  dom.btnBackStep1 = document.getElementById('btn-back-step1');
  dom.sampleSelect = document.getElementById('sample-select');
  dom.themeToggleBtn = document.getElementById('theme-toggle-btn');
  dom.resetAllBtn = document.getElementById('reset-all-btn');

  // Upload zones
  dom.dropzoneTemplate = document.getElementById('dropzone-template');
  dom.templateFileInput = document.getElementById('template-file-input');
  dom.templateEmpty = document.getElementById('template-dropzone-empty');
  dom.templatePreview = document.getElementById('template-dropzone-preview');
  dom.templateThumb = document.getElementById('template-thumb-img');
  dom.templateDimBadge = document.getElementById('template-dim-badge');
  dom.clearTemplateBtn = document.getElementById('clear-template-btn');

  dom.dropzoneSource = document.getElementById('dropzone-source');
  dom.sourceFileInput = document.getElementById('source-file-input');
  dom.sourceEmpty = document.getElementById('source-dropzone-empty');
  dom.sourcePreview = document.getElementById('source-dropzone-preview');
  dom.sourceThumb = document.getElementById('source-thumb-img');
  dom.sourceDimBadge = document.getElementById('source-dim-badge');
  dom.clearSourceBtn = document.getElementById('clear-source-btn');

  // Step 1 controls
  dom.sliderContrastLow = document.getElementById('slider-contrast-low');
  dom.inputContrastLow = document.getElementById('input-contrast-low');
  dom.sliderContrastHigh = document.getElementById('slider-contrast-high');
  dom.inputContrastHigh = document.getElementById('input-contrast-high');
  dom.sliderMinContrast = document.getElementById('slider-min-contrast');
  dom.inputMinContrast = document.getElementById('input-min-contrast');
  dom.sliderMinContLen = document.getElementById('slider-min-cont-len');
  dom.inputMinContLen = document.getElementById('input-min-cont-len');
  dom.selectUsePolarity = document.getElementById('select-use-polarity');
  dom.contrastValidation = document.getElementById('contrast-validation-msg');
  dom.btnAutoContrast = document.getElementById('btn-auto-contrast');

  dom.sliderAngleStart = document.getElementById('slider-angle-start');
  dom.inputAngleStart = document.getElementById('input-angle-start');
  dom.sliderAngleExtent = document.getElementById('slider-angle-extent');
  dom.inputAngleExtent = document.getElementById('input-angle-extent');
  dom.sliderAngleStep = document.getElementById('slider-angle-step');
  dom.inputAngleStep = document.getElementById('input-angle-step');
  dom.angleValidation = document.getElementById('angle-validation-msg');

  dom.levelRadios = document.querySelectorAll('input[name="num_levels"]');
  dom.autoExtractToggle = document.getElementById('auto-extract-toggle');
  dom.btnExtract = document.getElementById('btn-extract-features');
  dom.extractSpinner = document.getElementById('extract-spinner');

  dom.extractQualityBadge = document.getElementById('extract-quality-badge');
  dom.extractCountVal = document.getElementById('extract-count-val');
  dom.extractTimeVal = document.getElementById('extract-time-val');
  dom.extractHintMsg = document.getElementById('extract-hint-msg');

  // Step 2 controls
  dom.sliderNumMatches = document.getElementById('slider-num-matches');
  dom.inputNumMatches = document.getElementById('input-num-matches');
  dom.sliderMinScore = document.getElementById('slider-min-score');
  dom.inputMinScore = document.getElementById('input-min-score');
  dom.sliderScaleMin = document.getElementById('slider-scale-min');
  dom.inputScaleMin = document.getElementById('input-scale-min');
  dom.sliderScaleMax = document.getElementById('slider-scale-max');
  dom.inputScaleMax = document.getElementById('input-scale-max');
  dom.selectSubpixel = document.getElementById('select-subpixel');
  dom.sliderMaxOverlap = document.getElementById('slider-max-overlap');
  dom.inputMaxOverlap = document.getElementById('input-max-overlap');
  dom.sliderGreedness = document.getElementById('slider-greedness');
  dom.inputGreedness = document.getElementById('input-greedness');
  dom.scaleValidation = document.getElementById('scale-validation-msg');

  dom.btnRunMatch = document.getElementById('btn-run-match');
  dom.matchSpinner = document.getElementById('match-spinner');
  dom.matchResultBadge = document.getElementById('match-result-badge');
  dom.matchCountVal = document.getElementById('match-count-val');
  dom.matchTimeVal = document.getElementById('match-time-val');

  // Canvas stage
  dom.viewportContainer = document.getElementById('viewport-container');
  dom.singleViewport = document.getElementById('single-viewport');
  dom.splitViewport = document.getElementById('split-viewport');
  dom.mainCanvas = document.getElementById('main-canvas');
  dom.canvasPlaceholder = document.getElementById('canvas-placeholder');
  dom.splitTemplateCanvas = document.getElementById('split-template-canvas');
  dom.splitSourceCanvas = document.getElementById('split-source-canvas');

  dom.viewModeTemplate = document.getElementById('view-mode-template');
  dom.viewModeMatch = document.getElementById('view-mode-match');
  dom.viewModeSplit = document.getElementById('view-mode-split');
  dom.toggleLayerOverlay = document.getElementById('toggle-layer-overlay');
  dom.btnRoiSelect = document.getElementById('btn-roi-select');
  dom.btnQuickRoi = document.getElementById('btn-quick-roi');

  dom.btnZoomIn = document.getElementById('btn-zoom-in');
  dom.btnZoomOut = document.getElementById('btn-zoom-out');
  dom.btnZoomFit = document.getElementById('btn-zoom-fit');
  dom.btnZoom100 = document.getElementById('btn-zoom-100');
  dom.zoomLevelText = document.getElementById('zoom-level-text');

  dom.btnExportImage = document.getElementById('btn-export-image');
  dom.btnExportJson = document.getElementById('btn-export-json');

  dom.probeCoordXy = document.getElementById('probe-coord-xy');
  dom.probeImageDim = document.getElementById('probe-image-dim');

  // Results sidebar
  dom.resultsTableBody = document.getElementById('results-table-body');
  dom.resultsCountPill = document.getElementById('results-count-pill');
  dom.statBestScore = document.getElementById('stat-best-score');
  dom.statMatchCount = document.getElementById('stat-match-count');
  dom.statLatency = document.getElementById('stat-latency');
  dom.toastContainer = document.getElementById('toast-container');
}

// ===================================================================
// Step Workflow Navigation
// ===================================================================
function setStep(stepNumber) {
  state.currentStep = stepNumber;
  if (stepNumber === 1) {
    dom.stepBtn1.classList.add('active');
    dom.stepBtn2.classList.remove('active');
    dom.step1Panel.classList.remove('hidden');
    dom.step2Panel.classList.add('hidden');
    if (state.currentViewMode !== 'split') {
      setViewMode('template');
    }
  } else {
    dom.stepBtn2.classList.add('active');
    dom.stepBtn1.classList.remove('active');
    dom.step2Panel.classList.remove('hidden');
    dom.step1Panel.classList.add('hidden');
    if (state.currentViewMode !== 'split') {
      setViewMode('match');
    }
  }
}

function setViewMode(mode) {
  state.currentViewMode = mode;
  dom.viewModeTemplate.classList.toggle('active', mode === 'template');
  dom.viewModeMatch.classList.toggle('active', mode === 'match');
  dom.viewModeSplit.classList.toggle('active', mode === 'split');

  if (mode === 'split') {
    dom.singleViewport.classList.add('hidden');
    dom.splitViewport.classList.remove('hidden');
    renderSplitCanvas();
  } else {
    dom.singleViewport.classList.remove('hidden');
    dom.splitViewport.classList.add('hidden');
    fitCanvasToStage();
  }
}

// ===================================================================
// Event Listeners
// ===================================================================
function initEventListeners() {
  // Step navigation
  dom.stepBtn1.addEventListener('click', () => setStep(1));
  dom.stepBtn2.addEventListener('click', () => setStep(2));
  dom.btnGotoStep2.addEventListener('click', () => setStep(2));
  dom.btnBackStep1.addEventListener('click', () => setStep(1));

  // View modes
  dom.viewModeTemplate.addEventListener('click', () => setViewMode('template'));
  dom.viewModeMatch.addEventListener('click', () => setViewMode('match'));
  dom.viewModeSplit.addEventListener('click', () => setViewMode('split'));
  dom.toggleLayerOverlay.addEventListener('change', (e) => {
    state.showOverlayLayer = e.target.checked;
    renderCurrentView();
  });

  // Zoom buttons
  dom.btnZoomIn.addEventListener('click', () => adjustZoom(1.2));
  dom.btnZoomOut.addEventListener('click', () => adjustZoom(1 / 1.2));
  dom.btnZoomFit.addEventListener('click', () => fitCanvasToStage());
  dom.btnZoom100.addEventListener('click', () => {
    state.zoom = 1.0;
    centerCanvas();
    renderCurrentView();
  });

  // Presets selector
  dom.sampleSelect.addEventListener('change', (e) => {
    if (e.target.value) {
      loadSampleDataset(e.target.value);
    }
  });

  // Theme toggle & reset
  dom.themeToggleBtn.addEventListener('click', () => {
    const isDark = document.body.classList.contains('theme-dark');
    setTheme(isDark ? 'light' : 'dark');
  });

  dom.resetAllBtn.addEventListener('click', resetAll);

  // File Upload Dropzones
  setupDropzone(dom.dropzoneTemplate, dom.templateFileInput, handleTemplateFile);
  setupDropzone(dom.dropzoneSource, dom.sourceFileInput, handleSourceFile);

  dom.clearTemplateBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearTemplate();
  });
  dom.clearSourceBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearSource();
  });

  // Global Paste Image support
  window.addEventListener('paste', handleGlobalPaste);

  // Config Param Two-Way Binding (Step 1: pat_config)
  bindControl(dom.sliderContrastLow, dom.inputContrastLow, 'contrast_low', onPatConfigChanged);
  bindControl(dom.sliderContrastHigh, dom.inputContrastHigh, 'contrast_high', onPatConfigChanged);
  bindControl(dom.sliderMinContrast, dom.inputMinContrast, 'min_contrast', onPatConfigChanged);
  bindControl(dom.sliderMinContLen, dom.inputMinContLen, 'min_cont_len', onPatConfigChanged);
  bindControl(dom.sliderAngleStart, dom.inputAngleStart, 'angle_start', onPatConfigChanged);
  bindControl(dom.sliderAngleExtent, dom.inputAngleExtent, 'angle_extent', onPatConfigChanged);
  bindControl(dom.sliderAngleStep, dom.inputAngleStep, 'angle_step', onPatConfigChanged);

  dom.selectUsePolarity.addEventListener('change', (e) => {
    state.patConfig.use_polarity = parseInt(e.target.value, 10);
    onPatConfigChanged();
  });

  dom.levelRadios.forEach((r) => {
    r.addEventListener('change', (e) => {
      state.patConfig.num_levels = parseInt(e.target.value, 10);
      onPatConfigChanged();
    });
  });

  // Step 1 Extraction Action
  dom.btnExtract.addEventListener('click', () => extractTemplateFeatures(false));
  dom.btnAutoContrast.addEventListener('click', estimateTemplateContrast);

  // Angle preset chips
  document.querySelectorAll('.preset-chip[data-angle-start]').forEach((chip) => {
    chip.addEventListener('click', () => {
      setParamValue('angle_start', parseFloat(chip.dataset.angleStart));
      setParamValue('angle_extent', parseFloat(chip.dataset.angleExtent));
      onPatConfigChanged();
    });
  });

  // Config Param Two-Way Binding (Step 2: match_config)
  bindControl(dom.sliderNumMatches, dom.inputNumMatches, 'numMatches', onMatchConfigChanged);
  bindControl(dom.sliderMinScore, dom.inputMinScore, 'minScore', onMatchConfigChanged);
  bindControl(dom.sliderScaleMin, dom.inputScaleMin, 'scale_min', onMatchConfigChanged);
  bindControl(dom.sliderScaleMax, dom.inputScaleMax, 'scale_max', onMatchConfigChanged);
  bindControl(dom.sliderMaxOverlap, dom.inputMaxOverlap, 'maxOverLap', onMatchConfigChanged);
  bindControl(dom.sliderGreedness, dom.inputGreedness, 'greedness', onMatchConfigChanged);

  dom.selectSubpixel.addEventListener('change', (e) => {
    state.matchConfig.subpixel = parseInt(e.target.value, 10);
    onMatchConfigChanged();
  });

  // Initialize Micro-Stepper [-] and [+] Buttons with Hold-to-Repeat
  initStepperButtons();

  // Scale preset chips
  document.querySelectorAll('.preset-chip[data-scale-min]').forEach((chip) => {
    chip.addEventListener('click', () => {
      setParamValue('scale_min', parseFloat(chip.dataset.scaleMin));
      setParamValue('scale_max', parseFloat(chip.dataset.scaleMax));
      onMatchConfigChanged();
    });
  });

  dom.btnRunMatch.addEventListener('click', runShapeMatch);

  // ROI Selection Buttons
  if (dom.btnRoiSelect) {
    dom.btnRoiSelect.addEventListener('click', toggleRoiMode);
  }
  if (dom.btnQuickRoi) {
    dom.btnQuickRoi.addEventListener('click', () => {
      const img = getActiveDisplayImage() || state.sourceImage || state.templateImage;
      if (!img) {
        showToast('请先加载全景图或示例场景后再进行框选', 'warning');
        return;
      }
      if (state.sourceImage && state.currentViewMode !== 'match') {
        setViewMode('match');
      }
      enableRoiMode();
    });
  }

  // Keyboard shortcut: Ctrl + Enter to run match / R to toggle ROI / Escape to cancel ROI
  window.addEventListener('keydown', (e) => {
    const isEditingText = ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName);

    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      if (state.currentStep === 1) {
        extractTemplateFeatures(false);
      } else {
        runShapeMatch();
      }
    } else if ((e.key === 'r' || e.key === 'R') && !isEditingText && !e.ctrlKey && !e.metaKey) {
      e.preventDefault();
      toggleRoiMode();
    } else if (e.key === 'Escape' && state.isRoiMode) {
      e.preventDefault();
      disableRoiMode();
    }
  });

  // Export buttons
  dom.btnExportImage.addEventListener('click', exportCanvasImage);
  dom.btnExportJson.addEventListener('click', exportResultsJson);
}

function toggleRoiMode() {
  if (state.isRoiMode) {
    disableRoiMode();
  } else {
    enableRoiMode();
  }
}

function enableRoiMode() {
  const img = getActiveDisplayImage() || state.sourceImage || state.templateImage;
  if (!img) {
    showToast('请先在画布中加载或上传图像后再进行框选', 'warning');
    return;
  }
  state.isRoiMode = true;
  if (dom.btnRoiSelect) dom.btnRoiSelect.classList.add('active');
  if (dom.singleViewport) dom.singleViewport.classList.add('is-roi-mode');
  showToast('已进入模板框选模式：在画面上按住鼠标左键拖动以框选目标区域', 'info');
}

function disableRoiMode() {
  state.isRoiMode = false;
  state.isSelectingRoi = false;
  if (dom.btnRoiSelect) dom.btnRoiSelect.classList.remove('active');
  if (dom.singleViewport) dom.singleViewport.classList.remove('is-roi-mode');
  renderCurrentView();
}

function setTheme(theme) {
  if (theme === 'light') {
    document.body.classList.remove('theme-dark');
    document.body.classList.add('theme-light');
  } else {
    document.body.classList.remove('theme-light');
    document.body.classList.add('theme-dark');
  }
  localStorage.setItem('shapematch_theme', theme);
}

// ===================================================================
// Dropzone & File Handling
// ===================================================================
function setupDropzone(zoneEl, inputEl, fileHandler) {
  zoneEl.addEventListener('click', () => inputEl.click());

  inputEl.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      fileHandler(e.target.files[0]);
    }
  });

  zoneEl.addEventListener('dragover', (e) => {
    e.preventDefault();
    zoneEl.classList.add('drag-over');
  });

  zoneEl.addEventListener('dragleave', () => {
    zoneEl.classList.remove('drag-over');
  });

  zoneEl.addEventListener('drop', (e) => {
    e.preventDefault();
    zoneEl.classList.remove('drag-over');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      fileHandler(e.dataTransfer.files[0]);
    }
  });
}

function handleTemplateFile(file) {
  if (!file.type.startsWith('image/')) {
    showToast('请选择有效的图片文件 (PNG/JPG/BMP)', 'warning');
    return;
  }
  state.templateFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    loadTemplateImageFromSrc(e.target.result);
  };
  reader.readAsDataURL(file);
}

function handleSourceFile(file) {
  if (!file.type.startsWith('image/')) {
    showToast('请选择有效的图片文件 (PNG/JPG/BMP)', 'warning');
    return;
  }
  state.sourceFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    loadSourceImageFromSrc(e.target.result);
  };
  reader.readAsDataURL(file);
}

function handleGlobalPaste(e) {
  const items = (e.clipboardData || e.originalEvent.clipboardData).items;
  for (const item of items) {
    if (item.kind === 'file' && item.type.startsWith('image/')) {
      const blob = item.getAsFile();
      if (state.currentStep === 1 || !state.templateImage) {
        handleTemplateFile(blob);
        showToast('已从剪贴板粘贴为 Template 模板', 'info');
      } else {
        handleSourceFile(blob);
        showToast('已从剪贴板粘贴为 Source 场景图', 'info');
      }
      break;
    }
  }
}

function loadTemplateImageFromSrc(src, autoEstimate = true) {
  const img = new Image();
  img.onload = async () => {
    state.templateImage = img;
    state.templateOriginalB64 = src;
    state.templateViewB64 = null;
    state.templateViewImage = null;
    const imgW = img.naturalWidth || img.width;
    const imgH = img.naturalHeight || img.height;
    state.templateDim = { width: imgW, height: imgH };

    // Update UI thumbnail
    dom.templateThumb.src = src;
    dom.templateDimBadge.textContent = `${imgW} × ${imgH}`;
    dom.templateEmpty.classList.add('hidden');
    dom.templatePreview.classList.remove('hidden');

    showToast(`模板已载入 (${imgW}×${imgH} 像素)`, 'info');
    if (state.currentViewMode !== 'split') {
      setViewMode('template');
    }
    fitCanvasToStage(img);

    if (autoEstimate) {
      await estimateTemplateContrast(true);
    } else {
      extractTemplateFeatures(true);
    }
  };
  img.src = src;
}

function loadSourceImageFromSrc(src) {
  const img = new Image();
  img.onload = () => {
    state.sourceImage = img;
    state.sourceOriginalB64 = src;
    state.matchViewB64 = null;
    state.matchViewImage = null;
    state.matches = [];
    const imgW = img.naturalWidth || img.width;
    const imgH = img.naturalHeight || img.height;
    state.sourceDim = { width: imgW, height: imgH };

    // Update UI thumbnail
    dom.sourceThumb.src = src;
    dom.sourceDimBadge.textContent = `${imgW} × ${imgH}`;
    dom.sourceEmpty.classList.add('hidden');
    dom.sourcePreview.classList.remove('hidden');

    showToast(`全景图已载入 (${imgW}×${imgH} 像素)`, 'info');
    if (state.currentViewMode !== 'split') {
      setViewMode('match');
    }
    fitCanvasToStage(img);
  };
  img.src = src;
}

function clearTemplate() {
  state.templateFile = null;
  state.templateImage = null;
  state.templateOriginalB64 = null;
  state.templateViewB64 = null;
  state.templateViewImage = null;
  state.templateDim = { width: 0, height: 0 };
  state.featureCount = 0;

  dom.templateEmpty.classList.remove('hidden');
  dom.templatePreview.classList.add('hidden');
  dom.templateThumb.src = '';
  dom.extractQualityBadge.textContent = '待提取';
  dom.extractQualityBadge.className = 'status-badge badge-neutral';
  dom.extractCountVal.textContent = '-- / 256';
  dom.extractTimeVal.textContent = '-- ms';

  renderCurrentView();
}

function clearSource() {
  state.sourceFile = null;
  state.sourceImage = null;
  state.sourceOriginalB64 = null;
  state.matchViewB64 = null;
  state.matchViewImage = null;
  state.sourceDim = { width: 0, height: 0 };
  state.matches = [];

  dom.sourceEmpty.classList.remove('hidden');
  dom.sourcePreview.classList.add('hidden');
  dom.sourceThumb.src = '';
  updateResultsTable([]);

  renderCurrentView();
}

function resetAll() {
  clearTemplate();
  clearSource();
  dom.sampleSelect.value = '';

  state.patConfig = { ...DEFAULT_PAT_CONFIG };
  state.matchConfig = { ...DEFAULT_MATCH_CONFIG };

  updateFormControls();
  validateAllConfigs();
  setStep(1);
  showToast('工作区已全部重置', 'info');
}

// ===================================================================
// Presets Loading
// ===================================================================
async function loadSampleDataset(sampleId) {
  try {
    showToast(`正在加载示例场景: ${sampleId}...`, 'info');
    const resp = await fetch(apiUrl(`/samples/${sampleId}/images`));
    if (!resp.ok) {
      throw new Error(`HTTP error ${resp.status}`);
    }
    const data = await resp.json();
    if (!data.success) {
      throw new Error(data.message || '加载示例失败');
    }

    // Set configs
    if (data.default_pat) {
      state.patConfig = { ...state.patConfig, ...data.default_pat };
    }
    if (data.default_match) {
      state.matchConfig = { ...state.matchConfig, ...data.default_match };
    }
    updateFormControls();
    validateAllConfigs();

    // Load template image
    loadTemplateImageFromSrc(data.model_image, false);
    loadSourceImageFromSrc(data.source_image);

    showToast('示例场景已载入！可直接在步骤一微调特征或进入步骤二匹配', 'success');
  } catch (err) {
    console.error(err);
    showToast(`加载示例场景失败: ${err.message}`, 'error');
  }
}

// ===================================================================
// Parameter Constraints, Steppers & Form Binding
// ===================================================================
const PARAM_STEP_CONFIG = {
  contrast_low: { step: 1, digits: 0, absMin: 1, absMax: 254 },
  contrast_high: { step: 1, digits: 0, absMin: 2, absMax: 255 },
  min_contrast: { step: 1, digits: 0, absMin: 0, absMax: 10 },
  min_cont_len: { step: 1, digits: 0, absMin: 1, absMax: 1000 },
  angle_start: { step: 1.0, digits: 1, absMin: -360.0, absMax: 360.0 },
  angle_extent: { step: 1.0, digits: 1, absMin: 0.0, absMax: 360.0 },
  angle_step: { step: 1.0, digits: 1, absMin: 0.0, absMax: 360.0 },
  numMatches: { step: 1, digits: 0, absMin: 1, absMax: 100 },
  minScore: { step: 0.01, digits: 2, absMin: 0.0, absMax: 1.0 },
  scale_min: { step: 0.05, digits: 2, absMin: 0.1, absMax: 5.0 },
  scale_max: { step: 0.05, digits: 2, absMin: 0.1, absMax: 5.0 },
  maxOverLap: { step: 0.01, digits: 2, absMin: 0.0, absMax: 1.0 },
  greedness: { step: 0.01, digits: 2, absMin: 0.0, absMax: 1.0 },
};

function setParamValue(key, rawVal) {
  if (isNaN(rawVal)) return;

  if (key === 'contrast_low') {
    const maxAllowed = Math.max(1, state.patConfig.contrast_high - 1);
    state.patConfig.contrast_low = Math.min(Math.max(1, Math.round(rawVal)), maxAllowed);
  } else if (key === 'contrast_high') {
    const minAllowed = state.patConfig.contrast_low + 1;
    state.patConfig.contrast_high = Math.max(minAllowed, Math.min(255, Math.round(rawVal)));
  } else if (key === 'min_contrast') {
    state.patConfig.min_contrast = Math.max(0, Math.min(10, Math.round(rawVal)));
  } else if (key === 'min_cont_len') {
    state.patConfig.min_cont_len = Math.max(1, Math.min(1000, Math.round(rawVal)));
  } else if (key === 'angle_start') {
    state.patConfig.angle_start = Math.max(-360.0, Math.min(360.0, parseFloat(rawVal.toFixed(1))));
    const maxExtent = Math.min(360.0, Math.max(0.0, 360.0 - state.patConfig.angle_start));
    if (state.patConfig.angle_extent > maxExtent) {
      state.patConfig.angle_extent = parseFloat(maxExtent.toFixed(1));
    }
  } else if (key === 'angle_extent') {
    const maxExtent = Math.min(360.0, Math.max(0.0, 360.0 - state.patConfig.angle_start));
    state.patConfig.angle_extent = Math.max(0.0, Math.min(maxExtent, parseFloat(rawVal.toFixed(1))));
  } else if (key === 'angle_step') {
    state.patConfig.angle_step = Math.max(0.0, Math.min(360.0, parseFloat(rawVal.toFixed(1))));
  } else if (key === 'numMatches') {
    state.matchConfig.numMatches = Math.max(1, Math.min(100, Math.round(rawVal)));
  } else if (key === 'minScore') {
    state.matchConfig.minScore = Math.max(0.0, Math.min(1.0, parseFloat(rawVal.toFixed(2))));
  } else if (key === 'scale_min') {
    const maxScale = state.matchConfig.scale_max;
    state.matchConfig.scale_min = Math.max(0.1, Math.min(maxScale, parseFloat(rawVal.toFixed(2))));
  } else if (key === 'scale_max') {
    const minScale = state.matchConfig.scale_min;
    state.matchConfig.scale_max = Math.max(minScale, Math.min(5.0, parseFloat(rawVal.toFixed(2))));
  } else if (key === 'maxOverLap') {
    state.matchConfig.maxOverLap = Math.max(0.0, Math.min(1.0, parseFloat(rawVal.toFixed(2))));
  } else if (key === 'greedness') {
    state.matchConfig.greedness = Math.max(0.0, Math.min(1.0, parseFloat(rawVal.toFixed(2))));
  }

  updateFormControls();
}

function stepParam(key, direction) {
  const meta = PARAM_STEP_CONFIG[key];
  if (!meta) return;

  const isMatch = ['numMatches', 'minScore', 'scale_min', 'scale_max', 'maxOverLap', 'greedness'].includes(key);
  const targetObj = isMatch ? state.matchConfig : state.patConfig;
  const currentVal = parseFloat(targetObj[key]);
  const delta = meta.step * direction;
  const rawNewVal = parseFloat((currentVal + delta).toFixed(meta.digits));

  setParamValue(key, rawNewVal);

  if (isMatch) {
    onMatchConfigChanged();
  } else {
    onPatConfigChanged();
  }
}

function bindControl(slider, input, key, callback) {
  slider.addEventListener('input', (e) => {
    setParamValue(key, parseFloat(e.target.value));
    callback();
  });

  input.addEventListener('change', (e) => {
    const val = parseFloat(e.target.value);
    if (!isNaN(val)) {
      setParamValue(key, val);
      callback();
    } else {
      updateFormControls();
    }
  });

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      input.blur();
    }
  });
}

function initStepperButtons() {
  document.querySelectorAll('.micro-step-btn').forEach((btn) => {
    const target = btn.dataset.stepTarget;
    const dir = parseInt(btn.dataset.stepDir, 10);
    let stepTimer = null;
    let stepInterval = null;

    const doStep = () => {
      stepParam(target, dir);
    };

    const stopStepping = () => {
      clearTimeout(stepTimer);
      clearInterval(stepInterval);
      stepTimer = null;
      stepInterval = null;
    };

    btn.addEventListener('mousedown', (e) => {
      if (e.button !== 0) return;
      e.preventDefault();
      doStep();
      stepTimer = setTimeout(() => {
        stepInterval = setInterval(doStep, 65);
      }, 280);
    });

    btn.addEventListener('touchstart', (e) => {
      e.preventDefault();
      doStep();
      stepTimer = setTimeout(() => {
        stepInterval = setInterval(doStep, 65);
      }, 280);
    }, { passive: false });

    btn.addEventListener('mouseup', stopStepping);
    btn.addEventListener('mouseleave', stopStepping);
    btn.addEventListener('touchend', stopStepping);
    btn.addEventListener('touchcancel', stopStepping);
  });
}

function getCleanPatConfig() {
  return {
    contrast_low: parseInt(state.patConfig.contrast_low, 10),
    contrast_high: parseInt(state.patConfig.contrast_high, 10),
    min_contrast: parseInt(state.patConfig.min_contrast, 10),
    min_cont_len: parseInt(state.patConfig.min_cont_len, 10),
    angle_start: parseFloat(state.patConfig.angle_start),
    angle_extent: parseFloat(state.patConfig.angle_extent),
    angle_step: parseFloat(state.patConfig.angle_step),
    num_levels: parseInt(state.patConfig.num_levels, 10),
    use_polarity: parseInt(state.patConfig.use_polarity, 10),
  };
}

function getCleanMatchConfig() {
  return {
    numMatches: parseInt(state.matchConfig.numMatches, 10),
    minScore: parseFloat(state.matchConfig.minScore),
    scale_min: parseFloat(state.matchConfig.scale_min),
    scale_max: parseFloat(state.matchConfig.scale_max),
    subpixel: parseInt(state.matchConfig.subpixel, 10),
    maxOverLap: parseFloat(state.matchConfig.maxOverLap),
    greedness: parseFloat(state.matchConfig.greedness),
  };
}

function updateFormControls() {
  // Operational-level coupled bounds
  const lowMax = Math.max(1, state.patConfig.contrast_high - 1);
  const highMin = state.patConfig.contrast_low + 1;
  const maxExtent = Math.min(360.0, Math.max(0.0, 360.0 - state.patConfig.angle_start));
  const scaleMinMax = state.matchConfig.scale_max;
  const scaleMaxMin = state.matchConfig.scale_min;

  // Pat config bounds
  dom.sliderContrastLow.max = lowMax;
  dom.inputContrastLow.max = lowMax;
  dom.sliderContrastHigh.min = highMin;
  dom.inputContrastHigh.min = highMin;

  dom.sliderAngleExtent.max = maxExtent;
  dom.inputAngleExtent.max = maxExtent;

  // Match config bounds
  dom.sliderScaleMin.max = scaleMinMax;
  dom.inputScaleMin.max = scaleMinMax;
  dom.sliderScaleMax.min = scaleMaxMin;
  dom.inputScaleMax.min = scaleMaxMin;

  // Set values
  dom.sliderContrastLow.value = state.patConfig.contrast_low;
  dom.inputContrastLow.value = state.patConfig.contrast_low;
  dom.sliderContrastHigh.value = state.patConfig.contrast_high;
  dom.inputContrastHigh.value = state.patConfig.contrast_high;
  dom.sliderMinContrast.value = state.patConfig.min_contrast;
  dom.inputMinContrast.value = state.patConfig.min_contrast;
  dom.sliderMinContLen.value = state.patConfig.min_cont_len;
  dom.inputMinContLen.value = state.patConfig.min_cont_len;
  dom.selectUsePolarity.value = state.patConfig.use_polarity;

  dom.sliderAngleStart.value = state.patConfig.angle_start;
  dom.inputAngleStart.value = state.patConfig.angle_start;
  dom.sliderAngleExtent.value = state.patConfig.angle_extent;
  dom.inputAngleExtent.value = state.patConfig.angle_extent;
  dom.sliderAngleStep.value = state.patConfig.angle_step;
  dom.inputAngleStep.value = state.patConfig.angle_step;

  dom.levelRadios.forEach((r) => {
    r.checked = parseInt(r.value, 10) === state.patConfig.num_levels;
  });

  dom.sliderNumMatches.value = state.matchConfig.numMatches;
  dom.inputNumMatches.value = state.matchConfig.numMatches;
  dom.sliderMinScore.value = state.matchConfig.minScore;
  dom.inputMinScore.value = state.matchConfig.minScore;
  dom.sliderScaleMin.value = state.matchConfig.scale_min;
  dom.inputScaleMin.value = state.matchConfig.scale_min;
  dom.sliderScaleMax.value = state.matchConfig.scale_max;
  dom.inputScaleMax.value = state.matchConfig.scale_max;
  dom.selectSubpixel.value = state.matchConfig.subpixel;
  dom.sliderMaxOverlap.value = state.matchConfig.maxOverLap;
  dom.inputMaxOverlap.value = state.matchConfig.maxOverLap;
  dom.sliderGreedness.value = state.matchConfig.greedness;
  dom.inputGreedness.value = state.matchConfig.greedness;

  // Update button enabled/disabled states for extreme bounds
  updateStepperButtonStates();
}

function updateStepperButtonStates() {
  document.querySelectorAll('.micro-step-btn').forEach((btn) => {
    const target = btn.dataset.stepTarget;
    const dir = parseInt(btn.dataset.stepDir, 10);

    if (target === 'contrast_low') {
      if (dir === -1) btn.disabled = state.patConfig.contrast_low <= 1;
      if (dir === 1) btn.disabled = state.patConfig.contrast_low >= state.patConfig.contrast_high - 1;
    } else if (target === 'contrast_high') {
      if (dir === -1) btn.disabled = state.patConfig.contrast_high <= state.patConfig.contrast_low + 1;
      if (dir === 1) btn.disabled = state.patConfig.contrast_high >= 255;
    } else if (target === 'min_contrast') {
      if (dir === -1) btn.disabled = state.patConfig.min_contrast <= 0;
      if (dir === 1) btn.disabled = state.patConfig.min_contrast >= 10;
    } else if (target === 'min_cont_len') {
      if (dir === -1) btn.disabled = state.patConfig.min_cont_len <= 1;
      if (dir === 1) btn.disabled = state.patConfig.min_cont_len >= 1000;
    } else if (target === 'angle_start') {
      if (dir === -1) btn.disabled = state.patConfig.angle_start <= -360.0;
      if (dir === 1) btn.disabled = state.patConfig.angle_start >= 360.0;
    } else if (target === 'angle_extent') {
      const maxExt = Math.min(360.0, Math.max(0.0, 360.0 - state.patConfig.angle_start));
      if (dir === -1) btn.disabled = state.patConfig.angle_extent <= 0.0;
      if (dir === 1) btn.disabled = state.patConfig.angle_extent >= maxExt;
    } else if (target === 'angle_step') {
      if (dir === -1) btn.disabled = state.patConfig.angle_step <= 0.0;
      if (dir === 1) btn.disabled = state.patConfig.angle_step >= 360.0;
    } else if (target === 'numMatches') {
      if (dir === -1) btn.disabled = state.matchConfig.numMatches <= 1;
      if (dir === 1) btn.disabled = state.matchConfig.numMatches >= 100;
    } else if (target === 'minScore') {
      if (dir === -1) btn.disabled = state.matchConfig.minScore <= 0.0;
      if (dir === 1) btn.disabled = state.matchConfig.minScore >= 1.0;
    } else if (target === 'scale_min') {
      if (dir === -1) btn.disabled = state.matchConfig.scale_min <= 0.1;
      if (dir === 1) btn.disabled = state.matchConfig.scale_min >= state.matchConfig.scale_max;
    } else if (target === 'scale_max') {
      if (dir === -1) btn.disabled = state.matchConfig.scale_max <= state.matchConfig.scale_min;
      if (dir === 1) btn.disabled = state.matchConfig.scale_max >= 5.0;
    } else if (target === 'maxOverLap') {
      if (dir === -1) btn.disabled = state.matchConfig.maxOverLap <= 0.0;
      if (dir === 1) btn.disabled = state.matchConfig.maxOverLap >= 1.0;
    } else if (target === 'greedness') {
      if (dir === -1) btn.disabled = state.matchConfig.greedness <= 0.0;
      if (dir === 1) btn.disabled = state.matchConfig.greedness >= 1.0;
    }
  });
}

function validateAllConfigs() {
  let patValid = true;
  let matchValid = true;

  // Operational limits keep values in sync, validation confirms validity
  if (state.patConfig.contrast_low >= state.patConfig.contrast_high) {
    dom.contrastValidation.textContent = '警告: 低阈值必须严格小于高阈值 (contrast_low < contrast_high)';
    dom.contrastValidation.classList.remove('hidden');
    patValid = false;
  } else {
    dom.contrastValidation.classList.add('hidden');
  }

  if (state.patConfig.angle_start + state.patConfig.angle_extent > 360.0 + 1e-6) {
    dom.angleValidation.textContent = '警告: angle_start + angle_extent 不得超过 360°';
    dom.angleValidation.classList.remove('hidden');
    patValid = false;
  } else {
    dom.angleValidation.classList.add('hidden');
  }

  if (state.matchConfig.scale_min > state.matchConfig.scale_max) {
    dom.scaleValidation.textContent = '警告: 最小尺度不得大于最大尺度 (scale_min ≤ scale_max)';
    dom.scaleValidation.classList.remove('hidden');
    matchValid = false;
  } else {
    dom.scaleValidation.classList.add('hidden');
  }

  return { patValid, matchValid };
}

function onPatConfigChanged() {
  const { patValid } = validateAllConfigs();
  if (!patValid) return;

  if (dom.autoExtractToggle.checked && state.templateImage) {
    clearTimeout(state.autoExtractTimer);
    state.autoExtractTimer = setTimeout(() => {
      extractTemplateFeatures(true);
    }, 280);
  }
}

function onMatchConfigChanged() {
  validateAllConfigs();
}

// ===================================================================
// Step 1: Feature Extraction API Call
// ===================================================================
async function estimateTemplateContrast(showSuccessToast = true) {
  if (!state.templateOriginalB64) {
    showToast('请先拖拽或上传 Template 模板图片', 'warning');
    return;
  }

  const label = dom.btnAutoContrast ? dom.btnAutoContrast.querySelector('span') : null;
  if (dom.btnAutoContrast) {
    dom.btnAutoContrast.disabled = true;
    dom.btnAutoContrast.setAttribute('aria-busy', 'true');
  }
  if (label) label.textContent = '分析中…';

  try {
    const resp = await fetch(apiUrl('/estimate-contrast-json'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_base64: state.templateOriginalB64 }),
    });
    const data = await resp.json();
    if (!resp.ok || !data.success) {
      throw new Error(data.message || `HTTP error ${resp.status}`);
    }

    state.patConfig.contrast_low = data.contrast_low;
    state.patConfig.contrast_high = data.contrast_high;
    updateFormControls();
    validateAllConfigs();
    if (showSuccessToast) {
      showToast(`已自动估算对比度阈值：Low ${data.contrast_low} / High ${data.contrast_high}`, 'success');
    }

    if (dom.autoExtractToggle && dom.autoExtractToggle.checked) {
      await extractTemplateFeatures(true);
    }
  } catch (err) {
    console.error(err);
    showToast(`自动估算对比度失败: ${err.message}`, 'error');
  } finally {
    if (dom.btnAutoContrast) {
      dom.btnAutoContrast.disabled = false;
      dom.btnAutoContrast.removeAttribute('aria-busy');
    }
    if (label) label.textContent = 'Auto';
  }
}

async function extractTemplateFeatures(isDebounced = false) {
  if (!state.templateOriginalB64) {
    if (!isDebounced) {
      showToast('请先拖拽或上传 Template 模板图片', 'warning');
    }
    return;
  }

  const { patValid } = validateAllConfigs();
  if (!patValid) {
    showToast('请先修正 pat_config 参数错误', 'warning');
    return;
  }

  state.isExtracting = true;
  dom.extractSpinner.classList.remove('hidden');

  try {
    const payload = {
      model_base64: state.templateOriginalB64,
      pat_config: getCleanPatConfig(),
    };

    const resp = await fetch(apiUrl('/extract-template-json'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await resp.json();
    dom.extractSpinner.classList.add('hidden');
    state.isExtracting = false;

    if (!data.success) {
      dom.extractQualityBadge.textContent = '点数不足 (<8)';
      dom.extractQualityBadge.className = 'status-badge badge-danger';
      dom.extractCountVal.textContent = '0 / 256';
      dom.extractTimeVal.textContent = `${data.duration_ms || 0} ms`;
      dom.extractHintMsg.textContent = data.message || '特征提取失败';
      showToast(data.message || '模板特征不足，请降低对比度阈值', 'warning');
      return;
    }

    state.featureCount = data.feature_count;
    state.extractDurationMs = data.duration_ms;
    state.templateViewB64 = data.model_view;

    // Update status badge
    if (data.feature_count >= 64) {
      dom.extractQualityBadge.textContent = '特征极佳 (Excellent)';
      dom.extractQualityBadge.className = 'status-badge badge-success';
    } else if (data.feature_count >= 20) {
      dom.extractQualityBadge.textContent = '特征良好 (Good)';
      dom.extractQualityBadge.className = 'status-badge badge-success';
    } else {
      dom.extractQualityBadge.textContent = '特征偏少 (Sparse)';
      dom.extractQualityBadge.className = 'status-badge badge-warning';
    }

    dom.extractCountVal.textContent = `${data.feature_count} / 256`;
    dom.extractTimeVal.textContent = `${data.duration_ms} ms`;
    dom.extractHintMsg.textContent = `已成功提取 ${data.feature_count} 个稀疏梯度方向特征点 (耗时 ${data.duration_ms}ms)`;

    // Load view image into an Image element for canvas rendering
    const viewImg = new Image();
    viewImg.onload = () => {
      state.templateViewImage = viewImg;
      renderCurrentView();
    };
    viewImg.src = data.model_view;

    if (!isDebounced) {
      showToast(`特征提取成功！获得 ${data.feature_count} 个方向特征点`, 'success');
    }
  } catch (err) {
    dom.extractSpinner.classList.add('hidden');
    state.isExtracting = false;
    console.error(err);
    showToast(`特征提取请求失败: ${err.message}`, 'error');
  }
}

// ===================================================================
// Step 2: Shape-Based Matching API Call
// ===================================================================
async function runShapeMatch() {
  if (!state.templateOriginalB64) {
    showToast('请先在步骤一上传 Template 模板', 'warning');
    setStep(1);
    return;
  }
  if (!state.sourceOriginalB64) {
    showToast('请先拖拽或上传待搜索的 Source 场景图片', 'warning');
    return;
  }

  const { patValid, matchValid } = validateAllConfigs();
  if (!patValid || !matchValid) {
    showToast('请修正参数验证错误后再执行匹配', 'warning');
    return;
  }

  state.isMatching = true;
  dom.btnRunMatch.disabled = true;
  dom.matchSpinner.classList.remove('hidden');

  try {
    const payload = {
      model_base64: state.templateOriginalB64,
      source_base64: state.sourceOriginalB64,
      pat_config: getCleanPatConfig(),
      match_config: getCleanMatchConfig(),
    };

    const resp = await fetch(apiUrl('/match-json'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await resp.json();
    dom.btnRunMatch.disabled = false;
    dom.matchSpinner.classList.add('hidden');
    state.isMatching = false;

    if (!data.success) {
      showToast(data.message || '匹配执行失败', 'error');
      return;
    }

    state.matches = data.matches || [];
    state.matchDurationMs = data.duration_ms;
    state.matchViewB64 = data.match_view;

    // Update Telemetry Box & Result Cards
    const matchCount = state.matches.length;
    dom.statMatchCount.textContent = matchCount;
    dom.statLatency.textContent = `${data.duration_ms} ms`;
    dom.matchCountVal.textContent = matchCount;
    dom.matchTimeVal.textContent = `${data.duration_ms} ms`;

    if (matchCount > 0) {
      const best = state.matches[0].score;
      dom.statBestScore.textContent = best.toFixed(3);
      dom.matchResultBadge.textContent = `命中 ${matchCount} 个`;
      dom.matchResultBadge.className = 'status-badge badge-success';
      showToast(`匹配完成！耗时 ${data.duration_ms}ms，找到 ${matchCount} 处匹配实例`, 'success');
    } else {
      dom.statBestScore.textContent = '--';
      dom.matchResultBadge.textContent = '无符合条件目标';
      dom.matchResultBadge.className = 'status-badge badge-warning';
      showToast('未找到符合得分阈值的目标，建议尝试降低 minScore 或放宽尺度/角度范围', 'warning');
    }

    updateResultsTable(state.matches);

    // Load match overlay image
    if (data.match_view) {
      const matchImg = new Image();
      matchImg.onload = () => {
        state.matchViewImage = matchImg;
        setViewMode('match');
        renderCurrentView();
      };
      matchImg.src = data.match_view;
    } else {
      state.matchViewImage = null;
      renderCurrentView();
    }
  } catch (err) {
    dom.btnRunMatch.disabled = false;
    dom.matchSpinner.classList.add('hidden');
    state.isMatching = false;
    console.error(err);
    showToast(`匹配请求异常: ${err.message}`, 'error');
  }
}

// ===================================================================
// Results Table Rendering & Interactivity
// ===================================================================
function updateResultsTable(matches) {
  dom.resultsTableBody.innerHTML = '';
  dom.resultsCountPill.textContent = `${matches.length} 项`;

  if (!matches || matches.length === 0) {
    const tr = document.createElement('tr');
    tr.className = 'table-empty-row';
    tr.innerHTML = `<td colspan="5" class="empty-cell">${state.sourceImage ? '未搜索到符合阈值的目标' : '尚未执行目标匹配'}</td>`;
    dom.resultsTableBody.appendChild(tr);
    return;
  }

  matches.forEach((m, idx) => {
    const tr = document.createElement('tr');
    tr.dataset.index = idx;
    tr.innerHTML = `
      <td><strong>#${m.rank}</strong></td>
      <td><span style="color: var(--accent-emerald); font-weight:600;">${m.score.toFixed(3)}</span></td>
      <td>(${m.cx.toFixed(1)}, ${m.cy.toFixed(1)})</td>
      <td>${m.angle >= 0 ? '+' : ''}${m.angle.toFixed(1)}°</td>
      <td>x${m.scale.toFixed(2)}</td>
    `;

    // Row hover interaction: highlight on canvas
    tr.addEventListener('mouseenter', () => {
      state.hoveredMatchIndex = idx;
      tr.classList.add('row-selected');
      renderCurrentView();
    });

    tr.addEventListener('mouseleave', () => {
      state.hoveredMatchIndex = -1;
      tr.classList.remove('row-selected');
      renderCurrentView();
    });

    // Row click interaction: pan & focus on canvas
    tr.addEventListener('click', () => {
      state.selectedMatchIndex = idx;
      focusOnMatchTarget(m);
    });

    dom.resultsTableBody.appendChild(tr);
  });
}

function focusOnMatchTarget(match) {
  if (state.currentViewMode !== 'match') {
    setViewMode('match');
  }
  const stageW = dom.singleViewport.clientWidth;
  const stageH = dom.singleViewport.clientHeight;

  state.zoom = Math.max(state.zoom, 1.5);
  state.panX = stageW / 2 - match.cx * state.zoom;
  state.panY = stageH / 2 - match.cy * state.zoom;

  renderCurrentView();
  showToast(`已定位聚焦到目标 #${match.rank} (${match.cx}, ${match.cy})`, 'info');
}

// ===================================================================
// Interactive Canvas Engine (Pan & Zoom & Overlay Drawing)
// ===================================================================
function initCanvasViewport() {
  const stage = dom.singleViewport;

  // Window resize observer
  window.addEventListener('resize', () => {
    renderCurrentView();
  });

  // Mouse wheel zoom
  stage.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect = stage.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const zoomFactor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const newZoom = Math.min(Math.max(state.zoom * zoomFactor, 0.05), 30.0);

    // Zoom centered around mouse pointer
    state.panX = mouseX - (mouseX - state.panX) * (newZoom / state.zoom);
    state.panY = mouseY - (mouseY - state.panY) * (newZoom / state.zoom);
    state.zoom = newZoom;

    renderCurrentView();
  }, { passive: false });

  // Mouse drag pan or ROI selection
  stage.addEventListener('mousedown', (e) => {
    if (e.button === 0) { // Left click
      const isRoiActive = state.isRoiMode || e.shiftKey;
      const img = getActiveDisplayImage();

      if (isRoiActive && img) {
        const rect = stage.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        const imgW = img.naturalWidth || img.width;
        const imgH = img.naturalHeight || img.height;
        const imgX = (mouseX - state.panX) / state.zoom;
        const imgY = (mouseY - state.panY) / state.zoom;

        state.isSelectingRoi = true;
        state.roiStart = {
          x: Math.max(0, Math.min(imgW, imgX)),
          y: Math.max(0, Math.min(imgH, imgY)),
        };
        state.roiEnd = { ...state.roiStart };
        renderCurrentView();
        return;
      }

      state.isPanning = true;
      state.startPan = { x: e.clientX - state.panX, y: e.clientY - state.panY };
      stage.classList.add('is-panning');
    } else if (e.button === 1) { // Middle click always pans
      state.isPanning = true;
      state.startPan = { x: e.clientX - state.panX, y: e.clientY - state.panY };
      stage.classList.add('is-panning');
    }
  });

  window.addEventListener('mousemove', (e) => {
    if (state.isSelectingRoi) {
      const img = getActiveDisplayImage();
      if (img) {
        const rect = stage.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        const imgW = img.naturalWidth || img.width;
        const imgH = img.naturalHeight || img.height;

        const imgX = (mouseX - state.panX) / state.zoom;
        const imgY = (mouseY - state.panY) / state.zoom;

        state.roiEnd = {
          x: Math.max(0, Math.min(imgW, imgX)),
          y: Math.max(0, Math.min(imgH, imgY)),
        };
        renderCurrentView();
      }
    } else if (state.isPanning) {
      state.panX = e.clientX - state.startPan.x;
      state.panY = e.clientY - state.startPan.y;
      renderCurrentView();
    }

    // Pixel inspector update
    updateCoordinateProbe(e);
  });

  window.addEventListener('mouseup', () => {
    if (state.isSelectingRoi) {
      state.isSelectingRoi = false;
      finishRoiCrop();
    }
    if (state.isPanning) {
      state.isPanning = false;
      stage.classList.remove('is-panning');
    }
  });
}

function finishRoiCrop() {
  const img = getActiveDisplayImage();
  if (!img) return;

  const minX = Math.round(Math.min(state.roiStart.x, state.roiEnd.x));
  const minY = Math.round(Math.min(state.roiStart.y, state.roiEnd.y));
  const maxX = Math.round(Math.max(state.roiStart.x, state.roiEnd.x));
  const maxY = Math.round(Math.max(state.roiStart.y, state.roiEnd.y));

  const cropW = maxX - minX;
  const cropH = maxY - minY;

  // Need at least 8x8 pixels for valid template
  if (cropW >= 8 && cropH >= 8) {
    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = cropW;
    cropCanvas.height = cropH;
    const cropCtx = cropCanvas.getContext('2d');
    cropCtx.drawImage(img, minX, minY, cropW, cropH, 0, 0, cropW, cropH);

    const croppedDataUrl = cropCanvas.toDataURL('image/png');

    disableRoiMode();
    showToast(`已拖动框选截取模板 (${cropW} × ${cropH} px)，正在自动估算高低对比度阈值...`, 'info');

    // Load as template, autoEstimate = true will trigger estimateTemplateContrast
    loadTemplateImageFromSrc(croppedDataUrl, true);
    setStep(1);
    if (state.currentViewMode !== 'split') {
      setViewMode('template');
    }
  } else {
    renderCurrentView();
  }
}

function adjustZoom(factor) {
  const stage = dom.singleViewport;
  const stageW = stage.clientWidth;
  const stageH = stage.clientHeight;
  const newZoom = Math.min(Math.max(state.zoom * factor, 0.05), 30.0);

  state.panX = stageW / 2 - (stageW / 2 - state.panX) * (newZoom / state.zoom);
  state.panY = stageH / 2 - (stageH / 2 - state.panY) * (newZoom / state.zoom);
  state.zoom = newZoom;

  renderCurrentView();
}

function centerCanvas() {
  const stage = dom.singleViewport;
  const img = getActiveDisplayImage();
  if (!img) return;

  const imgW = img.naturalWidth || img.width;
  const imgH = img.naturalHeight || img.height;
  if (!imgW || !imgH) return;

  const stageW = stage.clientWidth || (stage.getBoundingClientRect && stage.getBoundingClientRect().width) || 600;
  const stageH = stage.clientHeight || (stage.getBoundingClientRect && stage.getBoundingClientRect().height) || 500;
  state.panX = Math.round((stageW - imgW * state.zoom) / 2);
  state.panY = Math.round((stageH - imgH * state.zoom) / 2);
}

function fitCanvasToStage(targetImg = null) {
  if (state.currentViewMode === 'split') {
    renderSplitCanvas();
    return;
  }

  const stage = dom.singleViewport;
  if (!stage) return;

  const isValidImg = targetImg && !(targetImg instanceof Event) && (targetImg.naturalWidth || targetImg.width);
  const img = isValidImg ? targetImg : getActiveDisplayImage();
  if (!img) {
    renderCurrentView();
    return;
  }

  const imgW = img.naturalWidth || img.width;
  const imgH = img.naturalHeight || img.height;
  if (!imgW || !imgH) {
    renderCurrentView();
    return;
  }

  const stageW = stage.clientWidth || (stage.getBoundingClientRect && stage.getBoundingClientRect().width) || 600;
  const stageH = stage.clientHeight || (stage.getBoundingClientRect && stage.getBoundingClientRect().height) || 500;
  const pad = 24; // comfortable margins around viewport
  const availW = Math.max(stageW - pad * 2, 40);
  const availH = Math.max(stageH - pad * 2, 40);

  const scaleX = availW / imgW;
  const scaleY = availH / imgH;
  
  // Calculate proportional scale to comfortably fit entirely within viewport
  let fitScale = Math.min(scaleX, scaleY);
  
  // Allow scaling up for small templates or down for large source images within application zoom range
  fitScale = Math.min(Math.max(fitScale, 0.01), 30.0);

  state.zoom = fitScale;
  state.panX = Math.round((stageW - imgW * state.zoom) / 2);
  state.panY = Math.round((stageH - imgH * state.zoom) / 2);

  renderCurrentView();
}

function getActiveDisplayImage() {
  if (state.currentViewMode === 'template') {
    return (state.showOverlayLayer && state.templateViewImage)
      ? state.templateViewImage
      : (state.templateImage || null);
  } else if (state.currentViewMode === 'match') {
    return (state.showOverlayLayer && state.matchViewImage)
      ? state.matchViewImage
      : (state.sourceImage || null);
  } else {
    return (state.showOverlayLayer && state.matchViewImage)
      ? state.matchViewImage
      : (state.sourceImage || state.templateImage || null);
  }
}

function updateCoordinateProbe(e) {
  const stage = dom.singleViewport;
  const rect = stage.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;

  if (mouseX < 0 || mouseX > rect.width || mouseY < 0 || mouseY > rect.height) {
    dom.probeCoordXy.textContent = 'X: --, Y: --';
    return;
  }

  const img = getActiveDisplayImage();
  if (!img) {
    dom.probeCoordXy.textContent = 'X: --, Y: --';
    dom.probeImageDim.textContent = '-- × --';
    return;
  }

  const imgW = img.naturalWidth || img.width;
  const imgH = img.naturalHeight || img.height;

  dom.probeImageDim.textContent = `${imgW} × ${imgH} px`;

  const imgX = Math.floor((mouseX - state.panX) / state.zoom);
  const imgY = Math.floor((mouseY - state.panY) / state.zoom);

  if (imgX >= 0 && imgX < imgW && imgY >= 0 && imgY < imgH) {
    dom.probeCoordXy.textContent = `X: ${imgX}, Y: ${imgY}`;
  } else {
    dom.probeCoordXy.textContent = 'X: --, Y: -- (越界)';
  }
}

function renderCurrentView() {
  dom.zoomLevelText.textContent = `${Math.round(state.zoom * 100)}%`;

  if (state.currentViewMode === 'split') {
    renderSplitCanvas();
  } else {
    renderMainCanvas();
  }
}

function renderMainCanvas() {
  const canvas = dom.mainCanvas;
  const stage = dom.singleViewport;
  if (!canvas || !stage) return;

  const dpr = window.devicePixelRatio || 1;
  const stageW = stage.clientWidth;
  const stageH = stage.clientHeight;

  canvas.width = stageW * dpr;
  canvas.height = stageH * dpr;
  canvas.style.width = `${stageW}px`;
  canvas.style.height = `${stageH}px`;

  const ctx = canvas.getContext('2d');
  ctx.resetTransform();
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, stageW, stageH);

  const img = getActiveDisplayImage();
  if (!img) {
    dom.canvasPlaceholder.classList.remove('hidden');
    return;
  }
  dom.canvasPlaceholder.classList.add('hidden');

  ctx.save();
  ctx.translate(state.panX, state.panY);
  ctx.scale(state.zoom, state.zoom);

  // Disable smoothing for sharp pixel inspection
  ctx.imageSmoothingEnabled = state.zoom < 2.0;

  // Draw base image
  ctx.drawImage(img, 0, 0);

  // If match mode, draw interactive highlight on hovered target
  if (state.currentViewMode === 'match' && state.hoveredMatchIndex >= 0) {
    const target = state.matches[state.hoveredMatchIndex];
    if (target && state.templateDim.width > 0) {
      drawMatchHighlight(ctx, target, state.templateDim);
    }
  }

  ctx.restore();

  // Draw ROI Drag-Selection Overlay
  if (state.isSelectingRoi) {
    const minX = Math.min(state.roiStart.x, state.roiEnd.x);
    const minY = Math.min(state.roiStart.y, state.roiEnd.y);
    const w = Math.abs(state.roiEnd.x - state.roiStart.x);
    const h = Math.abs(state.roiEnd.y - state.roiStart.y);

    const screenX = minX * state.zoom + state.panX;
    const screenY = minY * state.zoom + state.panY;
    const screenW = w * state.zoom;
    const screenH = h * state.zoom;

    // Direct draw in stage canvas coordinates
    ctx.save();
    ctx.fillStyle = 'rgba(6, 182, 212, 0.22)';
    ctx.fillRect(screenX, screenY, screenW, screenH);

    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(screenX, screenY, screenW, screenH);

    ctx.strokeStyle = 'rgba(15, 23, 42, 0.6)';
    ctx.lineWidth = 1;
    ctx.setLineDash([]);
    ctx.strokeRect(screenX, screenY, screenW, screenH);

    // Dimension HUD badge
    const badgeText = `${Math.round(w)} × ${Math.round(h)} px`;
    ctx.font = '600 11px JetBrains Mono, monospace';
    const textWidth = ctx.measureText(badgeText).width;
    const badgeW = textWidth + 14;
    const badgeH = 22;
    const badgeX = Math.max(8, Math.min(stageW - badgeW - 8, screenX));
    const badgeY = screenY - badgeH - 6 >= 6 ? screenY - badgeH - 6 : screenY + screenH + 6;

    ctx.fillStyle = 'rgba(15, 23, 42, 0.9)';
    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(badgeX, badgeY, badgeW, badgeH, 4);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#38bdf8';
    ctx.fillText(badgeText, badgeX + 7, badgeY + 15);
    ctx.restore();
  }
}

function drawMatchHighlight(ctx, match, tmplDim) {
  ctx.save();
  ctx.translate(match.cx, match.cy);
  ctx.rotate((-match.angle * Math.PI) / 180);
  ctx.scale(match.scale, match.scale);

  const hw = (tmplDim.width - 1) / 2;
  const hh = (tmplDim.height - 1) / 2;

  // Highlight glowing bounding rect
  ctx.strokeStyle = '#06b6d4';
  ctx.lineWidth = 3 / (state.zoom * match.scale);
  ctx.strokeRect(-hw, -hh, tmplDim.width, tmplDim.height);

  ctx.fillStyle = 'rgba(6, 182, 212, 0.2)';
  ctx.fillRect(-hw, -hh, tmplDim.width, tmplDim.height);

  ctx.restore();
}

function renderSplitCanvas() {
  // Render Left: Template
  if (dom.splitTemplateCanvas && state.templateImage) {
    const canvas = dom.splitTemplateCanvas;
    const box = document.getElementById('pane-template-box');
    const dpr = window.devicePixelRatio || 1;
    canvas.width = box.clientWidth * dpr;
    canvas.height = box.clientHeight * dpr;
    canvas.style.width = `${box.clientWidth}px`;
    canvas.style.height = `${box.clientHeight}px`;

    const ctx = canvas.getContext('2d');
    ctx.resetTransform();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, box.clientWidth, box.clientHeight);

    const img = state.showOverlayLayer && state.templateViewImage
      ? state.templateViewImage
      : state.templateImage;

    const imgW = img.naturalWidth || img.width;
    const imgH = img.naturalHeight || img.height;
    if (imgW && imgH) {
      const scale = Math.min(box.clientWidth / imgW, box.clientHeight / imgH) * 0.9;
      const px = (box.clientWidth - imgW * scale) / 2;
      const py = (box.clientHeight - imgH * scale) / 2;

      ctx.drawImage(img, px, py, imgW * scale, imgH * scale);
    }
  }

  // Render Right: Match
  if (dom.splitSourceCanvas && state.sourceImage) {
    const canvas = dom.splitSourceCanvas;
    const box = document.getElementById('pane-source-box');
    const dpr = window.devicePixelRatio || 1;
    canvas.width = box.clientWidth * dpr;
    canvas.height = box.clientHeight * dpr;
    canvas.style.width = `${box.clientWidth}px`;
    canvas.style.height = `${box.clientHeight}px`;

    const ctx = canvas.getContext('2d');
    ctx.resetTransform();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, box.clientWidth, box.clientHeight);

    const img = state.showOverlayLayer && state.matchViewImage
      ? state.matchViewImage
      : state.sourceImage;

    const imgW = img.naturalWidth || img.width;
    const imgH = img.naturalHeight || img.height;
    if (imgW && imgH) {
      const scale = Math.min(box.clientWidth / imgW, box.clientHeight / imgH) * 0.95;
      const px = (box.clientWidth - imgW * scale) / 2;
      const py = (box.clientHeight - imgH * scale) / 2;

      ctx.drawImage(img, px, py, imgW * scale, imgH * scale);
    }
  }
}

// ===================================================================
// Export Functions
// ===================================================================
function exportCanvasImage() {
  const img = getActiveDisplayImage();
  if (!img) {
    showToast('暂无可视图像可供导出', 'warning');
    return;
  }

  const link = document.createElement('a');
  link.download = `shapematch_${state.currentViewMode}_${Date.now()}.png`;
  link.href = img.src;
  link.click();
  showToast('标注图像下载已启动', 'success');
}

function exportResultsJson() {
  if (!state.matches || state.matches.length === 0) {
    showToast('尚无匹配结果数据可导出', 'warning');
    return;
  }

  const report = {
    export_time: new Date().toISOString(),
    pat_config: getCleanPatConfig(),
    match_config: getCleanMatchConfig(),
    duration_ms: state.matchDurationMs,
    match_count: state.matches.length,
    matches: state.matches,
  };

  const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.download = `match_results_${Date.now()}.json`;
  link.href = url;
  link.click();
  URL.revokeObjectURL(url);
  showToast('匹配结果 JSON 已导出', 'success');
}

// ===================================================================
// Toast Notification Utility
// ===================================================================
function showToast(message, type = 'info') {
  if (!dom.toastContainer) return;

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;

  let iconSvg = '';
  if (type === 'success') {
    iconSvg = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>';
  } else if (type === 'error') {
    iconSvg = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>';
  } else if (type === 'warning') {
    iconSvg = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>';
  } else {
    iconSvg = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>';
  }

  toast.innerHTML = `${iconSvg}<span>${message}</span>`;
  dom.toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(12px) scale(0.95)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}
