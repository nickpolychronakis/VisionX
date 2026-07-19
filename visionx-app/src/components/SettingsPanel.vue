<script setup lang="ts">
import { ref, computed } from "vue";

interface Settings {
  confidence: number;
  imgsz: number;
  outputDir: string;
  searchPrompts: string[];
  stride: number;
  parallel: number;
  halfPrecision: boolean;
  stitch: boolean;
  plates: boolean;
  faces: boolean;
}

const props = defineProps<{
  videoResolution?: number; // Max resolution of selected video(s)
}>();

const emit = defineEmits<{
  (e: "pick-output-dir"): void;
  (e: "clear-output-dir"): void;
}>();

const settings = defineModel<Settings>({ required: true });

const isExpanded = ref(false);
const showAdvanced = ref(false);
const newPrompt = ref("");

const outputDirName = computed(() => {
  const d = settings.value.outputDir;
  if (!d) return "";
  return d.split(/[\\/]/).pop() || d;
});

// Dynamic imgsz options based on video resolution
const imgszOptions = computed(() => {
  const maxRes = props.videoResolution || 640;
  const options = [
    { value: 320, label: "320px - Ταχύτατο" },
    { value: 480, label: "480px - Γρήγορο" },
    { value: 640, label: "640px - Κανονικό" },
    { value: 960, label: "960px - Υψηλή" },
    { value: 1280, label: "1280px - Πολύ υψηλή" },
    { value: 1920, label: "1920px - Πλήρης HD" },
  ];
  // Only show options up to the video resolution
  return options.filter((o) => o.value <= Math.max(maxRes, 320));
});

// Default label for imgsz
const imgszLabel = computed(() => {
  const maxRes = props.videoResolution || 0;
  if (maxRes > 0 && settings.value.imgsz >= maxRes) {
    return `${settings.value.imgsz}px (πλήρης)`;
  }
  return `${settings.value.imgsz}px`;
});

function addPrompt() {
  if (newPrompt.value.trim()) {
    settings.value.searchPrompts.push(newPrompt.value.trim());
    newPrompt.value = "";
  }
}

function removePrompt(index: number) {
  settings.value.searchPrompts.splice(index, 1);
}
</script>

<template>
  <div class="settings-panel card">
    <button class="toggle-btn" @click="isExpanded = !isExpanded">
      <span>Ρυθμίσεις</span>
      <span class="arrow" :class="{ expanded: isExpanded }">▼</span>
    </button>

    <div v-if="isExpanded" class="settings-content">
      <!-- Custom Search Prompts (most important - first) -->
      <div class="setting-row">
        <label>
          <span class="label-text">Τι ψάχνετε στο βίντεο;</span>
        </label>
        <p class="setting-hint">
          Η ανάλυση εντοπίζει πάντα ανθρώπους και οχήματα όλων των τύπων.
          Ό,τι γράψετε εδώ ΔΕΝ αλλάζει την ανίχνευση — επισημαίνει στα
          αποτελέσματα όσα ταιριάζουν (π.χ. «λευκό αυτοκίνητο», «άτομο με
          κόκκινη μπλούζα»). Ελληνικά ή αγγλικά.
        </p>
        <div class="prompts-input">
          <input
            type="text"
            v-model="newPrompt"
            placeholder="π.χ. λευκό αυτοκίνητο, σκύλος, ποδήλατο"
            @keyup.enter="addPrompt"
          />
          <button class="secondary" @click="addPrompt">Προσθήκη</button>
        </div>
        <div v-if="settings.searchPrompts.length > 0" class="prompts-list">
          <span
            v-for="(prompt, index) in settings.searchPrompts"
            :key="index"
            class="prompt-tag"
          >
            {{ prompt }}
            <button @click="removePrompt(index)">×</button>
          </span>
        </div>
      </div>

      <!-- Image Size -->
      <div class="setting-row">
        <label>
          <span class="label-text">Ποιότητα Ανάλυσης</span>
          <span class="label-value">{{ imgszLabel }}</span>
        </label>
        <select v-model.number="settings.imgsz">
          <option
            v-for="opt in imgszOptions"
            :key="opt.value"
            :value="opt.value"
          >
            {{ opt.label }}
          </option>
        </select>
        <p class="setting-hint">
          Κάθε καρέ συρρικνώνεται σε αυτό το μέγεθος πριν αναλυθεί.
          Μεγαλύτερη τιμή = καλύτερη αναγνώριση μικρών αντικειμένων, αλλά πιο αργή επεξεργασία.
          <template v-if="videoResolution && videoResolution > 0">
            <br/>Το βίντεό σας έχει ανάλυση {{ videoResolution }}px.
          </template>
        </p>
      </div>

      <!-- Confidence -->
      <div class="setting-row">
        <label>
          <span class="label-text">Βεβαιότητα Αναγνώρισης</span>
          <span class="label-value">{{ (settings.confidence * 100).toFixed(0) }}%</span>
        </label>
        <input
          type="range"
          v-model.number="settings.confidence"
          min="0.1"
          max="1"
          step="0.05"
        />
        <p class="setting-hint">
          {{ settings.confidence >= 0.7
            ? "Υψηλή βεβαιότητα — εμφανίζονται μόνο τα αντικείμενα που η AI είναι πολύ σίγουρη. Λιγότερα λάθη, αλλά μπορεί να χαθούν κάποια."
            : settings.confidence >= 0.4
              ? "Μέτρια βεβαιότητα — καλή ισορροπία μεταξύ ακρίβειας και κάλυψης."
              : "Χαμηλή βεβαιότητα — εμφανίζονται περισσότερα αντικείμενα, αλλά αυξάνονται και τα λανθασμένα αποτελέσματα."
          }}
        </p>
      </div>

      <!-- Analysis features (Phase A-Γ) -->
      <div class="setting-row">
        <label><span class="label-text">Λειτουργίες Ανάλυσης</span></label>
        <label class="switch-row">
          <input type="checkbox" v-model="settings.stitch" />
          <span>Ένωση διαδρομών — κάθε αντικείμενο μία φορά (όχι διπλές εγγραφές)</span>
        </label>
        <label class="switch-row">
          <input type="checkbox" v-model="settings.plates" />
          <span>Αυτόματη ανάγνωση πινακίδων στα οχήματα</span>
        </label>
        <label class="switch-row">
          <input type="checkbox" v-model="settings.faces" />
          <span>Εξαγωγή καθαρών λήψεων προσώπου (χωρίς αναγνώριση ταυτότητας)</span>
        </label>
      </div>

      <!-- Output directory -->
      <div class="setting-row">
        <label><span class="label-text">Φάκελος Αποθήκευσης</span></label>
        <div class="dir-row">
          <span class="dir-name" :title="settings.outputDir">
            {{ outputDirName || "Δίπλα στο βίντεο (προεπιλογή)" }}
          </span>
          <button class="secondary small" @click="emit('pick-output-dir')">Επιλογή…</button>
          <button v-if="settings.outputDir" class="secondary small" @click="emit('clear-output-dir')">✕</button>
        </div>
      </div>

      <!-- Advanced (collapsed) -->
      <div class="setting-row">
        <button class="advanced-toggle" @click="showAdvanced = !showAdvanced">
          <span class="arrow" :class="{ expanded: showAdvanced }">▸</span>
          Προχωρημένες ρυθμίσεις (ταχύτητα)
        </button>
      </div>
      <template v-if="showAdvanced">
        <div class="setting-row">
          <label>
            <span class="label-text">Βήμα καρέ (stride)</span>
            <span class="label-value">κάθε {{ settings.stride }}ο καρέ</span>
          </label>
          <input type="range" v-model.number="settings.stride" min="1" max="10" step="1" />
          <p class="setting-hint">
            Αναλύεται 1 στα Ν καρέ. Μεγαλύτερο = ταχύτερο, αλλά μπορεί να χαθούν
            σύντομες εμφανίσεις. Προτείνεται 1 για ασφάλεια.
          </p>
        </div>
        <div class="setting-row">
          <label>
            <span class="label-text">Παράλληλα βίντεο</span>
            <span class="label-value">{{ settings.parallel }}</span>
          </label>
          <input type="range" v-model.number="settings.parallel" min="1" max="8" step="1" />
          <p class="setting-hint">
            Επεξεργασία πολλών βίντεο ταυτόχρονα. Ταχύτερο σε δυνατά μηχανήματα,
            αλλά χρησιμοποιεί περισσότερη μνήμη/GPU.
          </p>
        </div>
        <div class="setting-row">
          <label class="switch-row">
            <input type="checkbox" v-model="settings.halfPrecision" />
            <span>FP16 — ταχύτερο σε κάρτες NVIDIA με Tensor Cores</span>
          </label>
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped>
.settings-panel {
  padding: 0;
}

.toggle-btn {
  width: 100%;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  background: transparent;
  color: var(--text-primary);
  font-size: 1rem;
  font-weight: 500;
}

.toggle-btn:hover {
  background: var(--bg-secondary);
}

.arrow {
  font-size: 0.8rem;
  transition: transform 0.2s ease;
}

.arrow.expanded {
  transform: rotate(180deg);
}

.settings-content {
  padding: 0 20px 20px;
  display: flex;
  flex-direction: column;
  max-height: 60vh;
  overflow-y: auto;
  gap: 20px;
}

.setting-row {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.setting-row label {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.label-text {
  font-size: 0.9rem;
}

.label-value {
  font-size: 0.85rem;
  color: var(--accent);
}

.setting-hint {
  font-size: 0.8rem;
  color: var(--text-secondary);
  margin: 0;
}

input[type="range"] {
  width: 100%;
  height: 6px;
  border-radius: 3px;
  background: var(--bg-secondary);
  -webkit-appearance: none;
}

input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--accent);
  cursor: pointer;
}

select {
  width: 100%;
  padding: 8px 12px;
  background: var(--bg-secondary);
  color: var(--text-primary);
  border: 1px solid var(--border);
  border-radius: 6px;
  font-size: 0.9rem;
}

.prompts-input {
  display: flex;
  gap: 8px;
}

.prompts-input input {
  flex: 1;
}

.prompts-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 8px;
}

.prompt-tag {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  background: var(--bg-secondary);
  border-radius: 20px;
  font-size: 0.85rem;
}

.prompt-tag button {
  background: transparent;
  color: var(--text-secondary);
  padding: 0;
  font-size: 1rem;
  line-height: 1;
}

.prompt-tag button:hover {
  color: var(--accent);
}

.switch-row {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 0.85rem;
  color: var(--text-secondary);
  cursor: pointer;
}

.switch-row input[type="checkbox"] {
  width: 16px;
  height: 16px;
  accent-color: var(--accent);
  flex-shrink: 0;
  cursor: pointer;
}

.dir-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.dir-name {
  flex: 1;
  font-size: 0.85rem;
  color: var(--text-secondary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.secondary.small {
  padding: 4px 12px;
  font-size: 0.8rem;
}

.advanced-toggle {
  background: transparent;
  color: var(--text-secondary);
  padding: 4px 0;
  font-size: 0.85rem;
  display: flex;
  align-items: center;
  gap: 6px;
}

.advanced-toggle:hover {
  color: var(--text-primary);
}

.advanced-toggle .arrow {
  transition: transform 0.2s ease;
}

.advanced-toggle .arrow.expanded {
  transform: rotate(90deg);
}
</style>
