<script setup lang="ts">
import { ref, onMounted, onUnmounted } from "vue";
import { open } from "@tauri-apps/plugin-dialog";
import { getCurrentWindow } from "@tauri-apps/api/window";

const emit = defineEmits<{
  filesSelected: [files: string[]];
}>();

const isDragging = ref(false);
let unlisten: (() => void) | null = null;

onMounted(async () => {
  const appWindow = getCurrentWindow();

  // Listen for file drop events from Tauri
  unlisten = await appWindow.onDragDropEvent((event) => {
    if (event.payload.type === 'enter' || event.payload.type === 'over') {
      isDragging.value = true;
    } else if (event.payload.type === 'drop') {
      isDragging.value = false;
      const paths = event.payload.paths;
      if (paths && paths.length > 0) {
        // Filter for video files
        const videoExtensions = ['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm', 'm4v', 'mpeg', 'mpg', '3gp', 'ts', 'mts'];
        const videoFiles = paths.filter(p => {
          const ext = p.split('.').pop()?.toLowerCase();
          return ext && videoExtensions.includes(ext);
        });
        if (videoFiles.length > 0) {
          emit("filesSelected", videoFiles);
        }
      }
    } else if (event.payload.type === 'leave') {
      isDragging.value = false;
    }
  });
});

onUnmounted(() => {
  if (unlisten) {
    unlisten();
  }
});

async function selectFiles() {
  try {
    const files = await open({
      multiple: true,
      filters: [
        {
          name: "Video",
          extensions: ["mp4", "avi", "mov", "mkv", "wmv", "flv", "webm", "m4v", "mpeg", "mpg", "3gp", "ts", "mts"],
        },
      ],
    });
    if (files) {
      emit("filesSelected", Array.isArray(files) ? files : [files]);
    }
  } catch (e) {
    console.error("Failed to open file dialog:", e);
  }
}

async function selectFolder() {
  try {
    const folder = await open({
      directory: true,
    });
    if (folder) {
      emit("filesSelected", [folder as string]);
    }
  } catch (e) {
    console.error("Failed to open folder dialog:", e);
  }
}
</script>

<template>
  <div class="file-selector card" :class="{ dragging: isDragging }">
    <div class="icon">
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
        <polyline points="17 8 12 3 7 8" />
        <line x1="12" y1="3" x2="12" y2="15" />
      </svg>
    </div>
    <p class="instruction">Σύρετε αρχεία βίντεο εδώ</p>
    <p class="or">ή</p>
    <div class="buttons">
      <button class="primary" @click="selectFiles">Επιλογή Αρχείων</button>
      <button class="secondary" @click="selectFolder">Επιλογή Φακέλου</button>
    </div>
    <p class="formats">Υποστηρίζονται: MP4, AVI, MOV, MKV, WMV, WebM και άλλα</p>
  </div>
</template>

<style scoped>
.file-selector {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 24px 20px;
  border: 2px dashed var(--border);
  transition: all 0.2s ease;
  flex-shrink: 0;
}

.file-selector.dragging {
  border-color: var(--accent);
  background: rgba(233, 69, 96, 0.1);
}

.icon {
  color: var(--text-secondary);
  margin-bottom: 10px;
}

.icon svg {
  width: 36px;
  height: 36px;
}

.instruction {
  font-size: 0.95rem;
  margin-bottom: 6px;
}

.or {
  color: var(--text-secondary);
  font-size: 0.8rem;
  margin-bottom: 10px;
}

.buttons {
  display: flex;
  gap: 10px;
  margin-bottom: 10px;
}

.formats {
  font-size: 0.7rem;
  color: var(--text-secondary);
}
</style>
