<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import LandingPage from "./components/LandingPage.vue";
import FileSelector from "./components/FileSelector.vue";
import SettingsPanel from "./components/SettingsPanel.vue";
import ProgressBar from "./components/ProgressBar.vue";
import ReportViewer from "./components/ReportViewer.vue";

// App state
type AppView = "landing" | "select" | "processing" | "results";
const currentView = ref<AppView>("landing");

// Selected files
const selectedFiles = ref<string[]>([]);

// Settings
const settings = ref({
  confidence: 0.65,
  stride: 1,
  halfPrecision: false,
  outputDir: "",
  searchPrompts: [] as string[],
});

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

// Reports
const reports = ref<string[]>([]);
const selectedReport = ref<string | null>(null);

// Event listener cleanup
let unlistenProgress: UnlistenFn | null = null;

// Computed
const canProcess = computed(() => selectedFiles.value.length > 0);

// Setup event listeners
onMounted(async () => {
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
    progress.value = {
      currentVideo: event.payload.video,
      currentFrame: event.payload.frame,
      totalFrames: event.payload.total_frames,
      videoIndex: event.payload.video_index,
      totalVideos: event.payload.total_videos,
      fps: event.payload.fps,
    };
  });
});

onUnmounted(() => {
  if (unlistenProgress) {
    unlistenProgress();
  }
});

// Methods
function onFilesSelected(files: string[]) {
  selectedFiles.value = files;
}

function removeFile(index: number) {
  selectedFiles.value.splice(index, 1);
}

async function startProcessing() {
  if (!canProcess.value) return;

  currentView.value = "processing";
  isProcessing.value = true;
  processingError.value = null;

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
        output_dir: settings.value.outputDir,
        search_prompts: settings.value.searchPrompts,
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
  <div class="app">
    <header class="header">
      <div class="header-content">
        <h1>VisionX</h1>
        <span class="subtitle">Ανίχνευση & Παρακολούθηση Αντικειμένων σε Βίντεο</span>
      </div>
      <button
        v-if="currentView !== 'landing'"
        class="info-btn"
        @click="currentView = 'landing'"
        title="Πληροφορίες"
      >
        ℹ️
      </button>
    </header>

    <main class="main">
      <!-- Error Message -->
      <div v-if="processingError" class="error-message card">
        <strong>Σφάλμα:</strong> {{ processingError }}
        <button class="dismiss-btn" @click="processingError = null">×</button>
      </div>

      <!-- Landing Page -->
      <LandingPage
        v-if="currentView === 'landing'"
        @start="currentView = 'select'"
      />

      <!-- File Selection View -->
      <div v-else-if="currentView === 'select'" class="view-select">
        <FileSelector @files-selected="onFilesSelected" />

        <div v-if="selectedFiles.length > 0" class="selected-files card">
          <h3>Επιλεγμένα Βίντεο ({{ selectedFiles.length }})</h3>
          <ul class="file-list">
            <li v-for="(file, index) in selectedFiles" :key="file">
              <span class="file-name">{{ file.split('/').pop() }}</span>
              <button class="remove-btn" @click="removeFile(index)">×</button>
            </li>
          </ul>
        </div>

        <SettingsPanel v-model="settings" />

        <div class="actions">
          <button
            class="primary start-btn"
            :disabled="!canProcess"
            @click="startProcessing"
          >
            Έναρξη Επεξεργασίας
          </button>
        </div>
      </div>

      <!-- Processing View -->
      <div v-else-if="currentView === 'processing'" class="view-processing">
        <ProgressBar
          :current-video="progress.currentVideo"
          :current-frame="progress.currentFrame"
          :total-frames="progress.totalFrames"
          :video-index="progress.videoIndex"
          :total-videos="progress.totalVideos"
          :fps="progress.fps"
        />
        <div class="actions">
          <button class="secondary" @click="cancelProcessing">Ακύρωση</button>
        </div>
      </div>

      <!-- Results View -->
      <div v-else-if="currentView === 'results'" class="view-results">
        <ReportViewer
          :reports="reports"
          v-model:selected="selectedReport"
        />
        <div class="actions">
          <button class="primary" @click="startNew">Επεξεργασία Νέων Βίντεο</button>
        </div>
      </div>
    </main>
  </div>
</template>

<style scoped>
.app {
  max-width: 900px;
  margin: 0 auto;
  padding: 20px;
}

.header {
  display: flex;
  justify-content: center;
  align-items: center;
  margin-bottom: 30px;
  position: relative;
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

.info-btn {
  position: absolute;
  right: 0;
  top: 50%;
  transform: translateY(-50%);
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
}

.view-select,
.view-processing,
.view-results {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.selected-files h3 {
  margin-bottom: 12px;
  font-size: 1rem;
}

.file-list {
  list-style: none;
  max-height: 200px;
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
</style>
