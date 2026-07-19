<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import AboutModal from "./components/AboutModal.vue";
import FileSelector from "./components/FileSelector.vue";
import SettingsPanel from "./components/SettingsPanel.vue";
import ProgressBar from "./components/ProgressBar.vue";
import ReportViewer from "./components/ReportViewer.vue";
import SetupWizard from "./components/SetupWizard.vue";

// App state — the app opens STRAIGHT into the workspace (2026 desktop UX:
// no landing page; identity/version/updates live in the About modal).
type AppView = "loading" | "setup" | "select" | "processing" | "results";
const currentView = ref<AppView>("loading");
const showAbout = ref(false);

// Theme — light is the default (user preference); dark is kept for
// night-time review of dark footage. Stored separately from settings so a
// future settings-reset never flips the theme under the user.
const THEME_KEY = "visionx.theme";
const theme = ref<"light" | "dark">(
  localStorage.getItem(THEME_KEY) === "dark" ? "dark" : "light",
);
watch(
  theme,
  (t) => {
    // styles.css switches every variable off this attribute
    document.documentElement.dataset.theme = t;
    localStorage.setItem(THEME_KEY, t);
  },
  { immediate: true },
);
function toggleTheme() {
  theme.value = theme.value === "light" ? "dark" : "light";
}

// Selected files
const selectedFiles = ref<string[]>([]);

// Settings — persisted to localStorage so they survive app restarts
// (previously volatile Vue refs, reset every launch).
const SETTINGS_KEY = "visionx.settings.v1";
const DEFAULT_SETTINGS = {
  confidence: 0.35,
  imgsz: 640,
  outputDir: "",
  // Structured filters (fixed choices) — free text removed by design.
  filterColors: [] as string[],
  filterTypes: [] as string[],
  // Advanced
  stride: 1,
  parallel: 1,
  halfPrecision: false,
  // Analysis features (Phase A-Γ) — all ON by default
  stitch: true,
  plates: true,
  faces: true,
};
function loadSettings() {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (raw) return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
  } catch (e) {
    console.error("Failed to load settings:", e);
  }
  return { ...DEFAULT_SETTINGS };
}
const settings = ref(loadSettings());

// Progress state
const progress = ref({
  currentVideo: "",
  currentFrame: 0,
  totalFrames: 0,
  videoIndex: 0,
  totalVideos: 0,
  fps: 0,
});

// Processing state
const isProcessing = ref(false);
const processingError = ref<string | null>(null);
const statusMessage = ref("");

// Reports
const reports = ref<string[]>([]);
const selectedReport = ref<string | null>(null);

// Event listener cleanup
let unlistenProgress: UnlistenFn | null = null;
let unlistenStatus: UnlistenFn | null = null;

// Computed
const canProcess = computed(() => selectedFiles.value.length > 0);

// Persist settings on every change (deep watch).
watch(settings, (val) => {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(val));
  } catch (e) {
    console.error("Failed to save settings:", e);
  }
}, { deep: true });

// Pick an output directory via the native folder dialog.
async function pickOutputDir() {
  const dir = await open({ directory: true, multiple: false });
  if (typeof dir === "string") settings.value.outputDir = dir;
}
function clearOutputDir() {
  settings.value.outputDir = "";
}

// Cross-video match mode (same event, multiple cameras) — per-run choice,
// deliberately NOT persisted: it depends on what the current videos are.
const matchMode = ref(false);

// Launch the interactive plate tool on a single video (plate-only mode).
async function runPlateTool() {
  if (selectedFiles.value.length === 0) return;
  try {
    await invoke("run_plate_tool", { video: selectedFiles.value[0] });
  } catch (e) {
    processingError.value = String(e);
  }
}

// Setup event listeners
onMounted(async () => {
  // Check if setup is needed
  try {
    const status = await invoke<{ needs_setup: boolean }>("check_setup");
    currentView.value = status.needs_setup ? "setup" : "select";
  } catch (e) {
    console.error("Failed to check setup:", e);
    currentView.value = "setup";
  }

  // Listen for progress events from backend
  unlistenProgress = await listen<{
    event_type: string;
    video: string;
    frame: number;
    total_frames: number;
    video_index: number;
    total_videos: number;
    fps: number;
  }>("progress", (event) => {
    // Clear status message when we start getting real progress
    if (event.payload.frame > 0) {
      statusMessage.value = "";
    }
    progress.value = {
      currentVideo: event.payload.video,
      currentFrame: event.payload.frame,
      totalFrames: event.payload.total_frames,
      videoIndex: event.payload.video_index,
      totalVideos: event.payload.total_videos,
      fps: event.payload.fps,
    };
  });

  // Listen for status events (model download, errors)
  unlistenStatus = await listen<{
    event_type: string;
    message: string;
  }>("status", (event) => {
    if (event.payload.event_type === "model_download") {
      statusMessage.value = event.payload.message;
    } else if (event.payload.event_type === "error") {
      processingError.value = event.payload.message;
    }
  });
});

onUnmounted(() => {
  if (unlistenProgress) unlistenProgress();
  if (unlistenStatus) unlistenStatus();
});

// Video resolution detection
const videoResolution = ref(0);

async function detectResolution(files: string[]) {
  if (files.length === 0) {
    videoResolution.value = 0;
    return;
  }
  try {
    const res = await invoke<number>("get_video_resolution", { files });
    videoResolution.value = res;
    // Auto-set imgsz to min(640, video resolution) as default
    if (res > 0 && res < settings.value.imgsz) {
      settings.value.imgsz = Math.max(320, Math.min(res, 640));
    }
  } catch {
    videoResolution.value = 0;
  }
}

// Methods
function onFilesSelected(files: string[]) {
  selectedFiles.value = files;
  detectResolution(files);
}

function removeFile(index: number) {
  selectedFiles.value.splice(index, 1);
}

async function startProcessing() {
  if (!canProcess.value) return;

  currentView.value = "processing";
  isProcessing.value = true;
  processingError.value = null;
  statusMessage.value = "";

  progress.value = {
    currentVideo: selectedFiles.value[0],
    currentFrame: 0,
    totalFrames: 0,
    videoIndex: 1,
    totalVideos: selectedFiles.value.length,
    fps: 0,
  };

  try {
    // Call the actual backend
    const result = await invoke<string[]>("process_videos", {
      files: selectedFiles.value,
      config: {
        confidence: settings.value.confidence,
        stride: settings.value.stride,
        half_precision: settings.value.halfPrecision,
        imgsz: settings.value.imgsz,
        parallel: settings.value.parallel,
        output_dir: settings.value.outputDir,
        filter_colors: settings.value.filterColors,
        filter_types: settings.value.filterTypes,
        stitch: settings.value.stitch,
        plates: settings.value.plates,
        faces: settings.value.faces,
        match_videos: matchMode.value,
      },
    });

    reports.value = result;
    // Auto-select first report to show preview
    if (result.length > 0) {
      selectedReport.value = result[0];
    }
    currentView.value = "results";
  } catch (error) {
    const errorStr = String(error);
    // Don't show error if it was cancelled
    if (!errorStr.toLowerCase().includes("cancelled")) {
      console.error("Processing failed:", error);
      processingError.value = errorStr;
    }
    currentView.value = "select";
  } finally {
    isProcessing.value = false;
  }
}

async function cancelProcessing() {
  try {
    await invoke("cancel_processing");
  } catch (e) {
    console.error("Failed to cancel:", e);
  }
  isProcessing.value = false;
  currentView.value = "select";
}

function startNew() {
  selectedFiles.value = [];
  reports.value = [];
  selectedReport.value = null;
  currentView.value = "select";
}
</script>

<template>
  <div class="app" :class="{ 'view-compact': currentView !== 'results' }">
    <header class="header">
      <div class="header-content">
        <h1>VisionX</h1>
        <span class="subtitle">Ανίχνευση & Παρακολούθηση Αντικειμένων σε Βίντεο</span>
      </div>
      <div class="header-actions" v-if="currentView !== 'setup' && !isProcessing">
        <button
          class="info-btn theme-btn"
          @click="toggleTheme"
          :title="theme === 'light' ? 'Σκούρο θέμα' : 'Φωτεινό θέμα'"
        >
          {{ theme === "light" ? "🌙" : "☀️" }}
        </button>
        <button
          class="info-btn"
          @click="showAbout = true"
          title="Σχετικά με το VisionX"
        >
          ℹ️
        </button>
      </div>
    </header>

    <Transition name="modal">
      <AboutModal v-if="showAbout" @close="showAbout = false" />
    </Transition>

    <main class="main">
      <!-- Loading -->
      <div v-if="currentView === 'loading'" class="view-loading">
        <div class="spinner"></div>
      </div>

      <!-- Setup Wizard -->
      <SetupWizard
        v-else-if="currentView === 'setup'"
        @complete="currentView = 'select'"
      />

      <!-- Error Message -->
      <div v-if="processingError" class="error-message card">
        <strong>Σφάλμα:</strong> {{ processingError }}
        <button class="dismiss-btn" @click="processingError = null">×</button>
      </div>

      <!-- File Selection View (the workspace home) -->
      <Transition name="view" mode="out-in">
      <div v-if="currentView === 'select'" class="view-select" key="select">
        <FileSelector @files-selected="onFilesSelected" />

        <div v-if="selectedFiles.length > 0" class="selected-files card card-teal">
          <h3>Επιλεγμένα Βίντεο ({{ selectedFiles.length }})</h3>
          <ul class="file-list">
            <li v-for="(file, index) in selectedFiles" :key="file">
              <span class="file-name">{{ file.split('/').pop() }}</span>
              <button class="remove-btn" @click="removeFile(index)">×</button>
            </li>
          </ul>
        </div>

        <SettingsPanel
          v-if="selectedFiles.length > 0"
          v-model="settings"
          :videoResolution="videoResolution"
          @pick-output-dir="pickOutputDir"
          @clear-output-dir="clearOutputDir"
        />

        <label v-if="selectedFiles.length > 1" class="match-toggle">
          <input type="checkbox" v-model="matchMode" />
          <span>Ίδιο συμβάν από πολλές κάμερες — αντιστοίχιση αντικειμένων
          μεταξύ των βίντεο (+ συγκεντρωτική αναφορά)</span>
        </label>

        <div v-if="selectedFiles.length > 0" class="actions">
          <button
            class="primary start-btn"
            :disabled="!canProcess"
            @click="startProcessing"
          >
            Έναρξη Επεξεργασίας
          </button>
          <button
            class="secondary plate-btn"
            :title="selectedFiles.length > 1 ? 'Χρησιμοποιεί το πρώτο βίντεο' : ''"
            @click="runPlateTool"
          >
            🚘 Ανάγνωση Πινακίδας
          </button>
        </div>
      </div>

      <!-- Processing View -->
      <div v-else-if="currentView === 'processing'" class="view-processing" key="processing">
        <ProgressBar
          :current-video="progress.currentVideo"
          :current-frame="progress.currentFrame"
          :total-frames="progress.totalFrames"
          :video-index="progress.videoIndex"
          :total-videos="progress.totalVideos"
          :fps="progress.fps"
          :status-message="statusMessage"
        />
        <div class="actions">
          <button class="secondary" @click="cancelProcessing">Ακύρωση</button>
        </div>
      </div>

      <!-- Results View -->
      <div v-else-if="currentView === 'results'" class="view-results" key="results">
        <ReportViewer
          :reports="reports"
          v-model:selected="selectedReport"
        />
        <div class="actions">
          <button class="primary" @click="startNew">Επεξεργασία Νέων Βίντεο</button>
        </div>
      </div>
      </Transition>
    </main>
  </div>
</template>

<style scoped>
.app {
  display: flex;
  flex-direction: column;
  height: 100%;
  padding: 20px;
}

.app.view-compact {
  max-width: 900px;
  margin: 0 auto;
}

.header {
  display: flex;
  justify-content: center;
  align-items: center;
  padding-bottom: 20px;
  position: relative;
  flex-shrink: 0;
}

.header-content {
  text-align: center;
}

.header h1 {
  font-size: 2rem;
  color: var(--accent);
  margin-bottom: 4px;
}

.subtitle {
  color: var(--text-secondary);
  font-size: 0.9rem;
}

.header-actions {
  position: absolute;
  right: 0;
  top: 50%;
  transform: translateY(-50%);
  display: flex;
  gap: 8px;
}

.info-btn {
  background: transparent;
  border: 1px solid var(--border);
  border-radius: 50%;
  width: 36px;
  height: 36px;
  font-size: 1rem;
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0;
  line-height: 1;
}

.info-btn:hover {
  background: var(--bg-secondary);
  border-color: var(--accent);
}

.main {
  display: flex;
  flex-direction: column;
  gap: 20px;
  flex: 1;
  min-height: 0;
}

.view-select {
  display: flex;
  flex-direction: column;
  gap: 20px;
  flex: 1;
  min-height: 0;
  overflow-y: auto;
}

/* Empty state: when the drop zone is the only content, center it in the
   window instead of leaving a void below. */
.view-select > :first-child:last-child {
  margin-block: auto;
}

.view-processing {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.view-results {
  display: flex;
  flex-direction: column;
  gap: 20px;
  flex: 1;
  min-height: 0;
  overflow: hidden;
}

.view-processing {
  justify-content: center;
  align-items: center;
}

.selected-files h3 {
  margin-bottom: 12px;
  font-size: 1rem;
}

.selected-files {
  flex-shrink: 0;
  max-height: 150px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.file-list {
  list-style: none;
  flex: 1;
  overflow-y: auto;
}

.file-list li {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  background: var(--bg-secondary);
  border-radius: 6px;
  margin-bottom: 6px;
}

.file-name {
  font-size: 0.9rem;
  color: var(--text-secondary);
}

.remove-btn {
  background: transparent;
  color: var(--text-secondary);
  padding: 2px 8px;
  font-size: 1.2rem;
}

.remove-btn:hover {
  color: var(--accent);
}

.actions {
  display: flex;
  justify-content: center;
  gap: 12px;
  margin-top: 10px;
}

.start-btn {
  padding: 14px 40px;
  font-size: 1rem;
}

.plate-btn {
  padding: 14px 24px;
  font-size: 1rem;
}

.match-toggle {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 0.9rem;
  color: var(--text-secondary);
  cursor: pointer;
  padding: 0 4px;
}

.match-toggle input[type="checkbox"] {
  width: 16px;
  height: 16px;
  accent-color: var(--accent);
  flex-shrink: 0;
}

.error-message {
  background: #ff4444;
  color: white;
  padding: 12px 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.dismiss-btn {
  background: transparent;
  color: white;
  font-size: 1.2rem;
  padding: 2px 8px;
}

.view-loading {
  display: flex;
  justify-content: center;
  align-items: center;
  flex: 1;
}

/* View transitions: subtle fade + rise between workspace states */
.view-enter-active,
.view-leave-active {
  transition: opacity 0.22s ease, transform 0.22s ease;
}

.view-enter-from {
  opacity: 0;
  transform: translateY(10px);
}

.view-leave-to {
  opacity: 0;
  transform: translateY(-6px);
}

.modal-enter-active,
.modal-leave-active {
  transition: opacity 0.2s ease;
}

.modal-enter-from,
.modal-leave-to {
  opacity: 0;
}

@media (prefers-reduced-motion: reduce) {
  .view-enter-active,
  .view-leave-active,
  .modal-enter-active,
  .modal-leave-active {
    transition: none;
  }
}

.spinner {
  width: 40px;
  height: 40px;
  border: 3px solid var(--border);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>
