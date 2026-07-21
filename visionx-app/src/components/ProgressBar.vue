<script setup lang="ts">
import { computed } from "vue";

const props = defineProps<{
  currentVideo: string;
  currentFrame: number;
  totalFrames: number;
  videoIndex: number;
  totalVideos: number;
  fps: number;
  statusMessage: string;
  // Live annotated frame (base64 JPEG) from the analysis — empty when the
  // preview is unavailable (e.g. parallel processing).
  liveFrame: string;
}>();

const framePercent = computed(() => {
  if (props.totalFrames === 0) return 0;
  return Math.round((props.currentFrame / props.totalFrames) * 100);
});

const overallPercent = computed(() => {
  if (props.totalVideos === 0) return 0;
  if (props.totalFrames === 0) return 0;
  const completedVideos = props.videoIndex - 1;
  const currentProgress = props.currentFrame / props.totalFrames;
  return Math.round(((completedVideos + currentProgress) / props.totalVideos) * 100);
});

const videoName = computed(() => {
  return props.currentVideo.split("/").pop() || props.currentVideo;
});

const estimatedTimeLeft = computed(() => {
  if (props.fps <= 0 || props.currentFrame <= 0) return "";
  const remaining = props.totalFrames - props.currentFrame;
  const seconds = Math.round(remaining / props.fps);
  if (seconds < 60) return `~${seconds}s`;
  if (seconds < 3600) return `~${Math.round(seconds / 60)}m`;
  const h = Math.floor(seconds / 3600);
  const m = Math.round((seconds % 3600) / 60);
  return `~${h}h ${m}m`;
});
</script>

<template>
  <div class="progress-container card">
    <div class="current-video">
      <span class="label">Επεξεργασία:</span>
      <span class="video-name">{{ videoName }}</span>
    </div>

    <!-- Status message (downloading model, etc.) -->
    <div v-if="statusMessage" class="loading-section">
      <div class="spinner"></div>
      <p>{{ statusMessage }}</p>
    </div>

    <!-- Loading indicator when totalFrames is 0 -->
    <div v-else-if="totalFrames === 0" class="loading-section">
      <div class="spinner"></div>
      <p>Αρχικοποίηση... Η φόρτωση του μοντέλου μπορεί να πάρει λίγο χρόνο.</p>
      <p class="hint">Η ανάλυση θα ξεκινήσει μόλις φορτωθεί το μοντέλο.</p>
    </div>

    <!-- Progress bars only shown when we have frame data -->
    <template v-else>
      <!-- Live detection preview: annotated frames streamed from the
           analysis (boxes, track IDs, confidences) -->
      <div v-if="liveFrame" class="live-preview">
        <img :src="'data:image/jpeg;base64,' + liveFrame" alt="Ζωντανή προεπισκόπηση ανάλυσης" />
        <span class="live-badge">● LIVE</span>
      </div>

      <!-- Current video progress -->
      <div class="progress-section">
        <div class="progress-header">
          <span>Πρόοδος Καρέ</span>
          <span>{{ currentFrame }} / {{ totalFrames }} ({{ framePercent }}%)</span>
        </div>
        <div class="progress-bar">
          <div class="progress-fill" :style="{ width: framePercent + '%' }"></div>
        </div>
      </div>

      <!-- Overall progress -->
      <div class="progress-section">
        <div class="progress-header">
          <span>Συνολική Πρόοδος</span>
          <span>Βίντεο {{ videoIndex }} από {{ totalVideos }} ({{ overallPercent }}%)</span>
        </div>
        <div class="progress-bar overall">
          <div class="progress-fill" :style="{ width: overallPercent + '%' }"></div>
        </div>
      </div>

      <!-- Stats -->
      <div class="stats">
        <div class="stat">
          <span class="stat-value">{{ fps.toFixed(1) }}</span>
          <span class="stat-label">frames/sec</span>
        </div>
        <div v-if="estimatedTimeLeft" class="stat">
          <span class="stat-value">{{ estimatedTimeLeft }}</span>
          <span class="stat-label">Remaining</span>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.progress-container {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.current-video {
  display: flex;
  align-items: center;
  gap: 8px;
}

.label {
  color: var(--text-secondary);
  font-size: 0.9rem;
}

.video-name {
  font-weight: 500;
  color: var(--accent);
}

.progress-section {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.progress-header {
  display: flex;
  justify-content: space-between;
  font-size: 0.85rem;
  color: var(--text-secondary);
}

.progress-bar {
  height: 8px;
  background: var(--bg-secondary);
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: var(--accent);
  border-radius: 4px;
  transition: width 0.3s ease;
}

.progress-bar.overall .progress-fill {
  background: var(--success);
}

.stats {
  display: flex;
  justify-content: center;
  gap: 40px;
  padding-top: 10px;
  border-top: 1px solid var(--border);
}

.stat {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.stat-value {
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--accent);
}

.stat-label {
  font-size: 0.75rem;
  color: var(--text-secondary);
  text-transform: uppercase;
}

.loading-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 40px 20px;
  text-align: center;
  /* The card fills the window during processing — keep the init spinner
     vertically centered in it instead of top-stuck. */
  margin-block: auto;
}

.spinner {
  width: 40px;
  height: 40px;
  border: 3px solid var(--border);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 16px;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.loading-section p {
  margin: 4px 0;
}

.loading-section .hint {
  font-size: 0.85rem;
  color: var(--text-secondary);
}
.live-preview {
  position: relative;
  margin-bottom: 16px;
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid var(--border);
  background: #000;
  /* Grow into whatever height the card offers (the card fills the window
     during processing) — the flex bound replaces the old 34vh cap, and
     min-height: 0 keeps Cancel reachable on small windows. */
  flex: 1;
  min-height: 120px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.live-preview img {
  display: block;
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}


.live-badge {
  position: absolute;
  top: 8px;
  left: 10px;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.5px;
  color: #fff;
  background: rgba(217, 43, 75, 0.85);
  border-radius: 6px;
  padding: 2px 8px;
}
</style>
