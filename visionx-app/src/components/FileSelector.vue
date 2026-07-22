<script setup lang="ts">
import { ref, onMounted, onUnmounted } from "vue";
import { open } from "@tauri-apps/plugin-dialog";
import { invoke } from "@tauri-apps/api/core";
import { getCurrentWindow } from "@tauri-apps/api/window";

const emit = defineEmits<{
  filesSelected: [files: string[]];
}>();

const isDragging = ref(false);
// Feedback when a picked/dropped folder yields no videos — otherwise the
// user's click appears to do nothing.
const folderNote = ref("");
let unlisten: (() => void) | null = null;

// Mirrors vision.py VIDEO_EXTENSIONS (source of truth; sync enforced by
// tests/test_consistency.py — the lists had silently diverged before).
const videoExtensions = ['mp4', 'm4v', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm', 'mpeg', 'mpg', 'ts', 'mts', 'm2ts', '3gp', 'asf', 'dav', 'bin'];

// Expand any directories among the paths into the video files inside them;
// keep direct video files as-is. The list always ends up holding individual
// files — a raw folder path used to reach OpenCV and fail mid-processing.
async function expandPaths(paths: string[]): Promise<string[]> {
  const files: string[] = [];
  for (const p of paths) {
    const ext = p.split('.').pop()?.toLowerCase();
    if (ext && videoExtensions.includes(ext) && !p.endsWith('/')) {
      files.push(p);
      continue;
    }
    try {
      files.push(...await invoke<string[]>("list_videos_in_dir", { dir: p }));
    } catch {
      // Not a directory and not a known video extension — ignore.
    }
  }
  return files;
}

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
        expandPaths(paths).then((videoFiles) => {
          folderNote.value = videoFiles.length > 0
            ? ""
            : "Δεν βρέθηκαν αρχεία βίντεο σε ό,τι αποθέσατε.";
          if (videoFiles.length > 0) emit("filesSelected", videoFiles);
        });
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
          // Same canonical list as videoExtensions above (one constant).
          extensions: videoExtensions,
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
      const videos = await invoke<string[]>("list_videos_in_dir", {
        dir: folder as string,
      });
      folderNote.value = videos.length > 0
        ? ""
        : "Ο φάκελος δεν περιέχει αρχεία βίντεο.";
      if (videos.length > 0) emit("filesSelected", videos);
    }
  } catch (e) {
    console.error("Failed to open folder dialog:", e);
    folderNote.value = "Αποτυχία ανάγνωσης του φακέλου.";
  }
}
</script>

<template>
  <div class="file-selector card card-blue" :class="{ dragging: isDragging }">
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
    <p v-if="folderNote" class="folder-note">{{ folderNote }}</p>
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

.folder-note {
  margin-top: 8px;
  font-size: 0.8rem;
  color: var(--danger);
}
</style>
