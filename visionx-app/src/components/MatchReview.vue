<script setup lang="ts">
// Match-review screen (user workflow): after a multi-camera analysis, the
// investigator SEES what was auto-matched (and can reject it — the
// algorithm does err) and what is merely suspected (and can accept it).
// The decisions then regenerate ONE combined report via vision.py
// --finalize-match. Human judgment always outranks the algorithm.
import { ref, onMounted } from "vue";
import { invoke } from "@tauri-apps/api/core";

const props = defineProps<{ reviewPath: string }>();
const emit = defineEmits<{ done: [reportPath: string]; skip: [] }>();

interface TrackInfo {
  class: string;
  first_seen: number;
  last_seen: number;
  static: boolean;
  plate: string | null;
  color: string | null;
  thumb: string | null;
}

interface Group {
  members: [string, number][];
  evidence: string | null;
  score: number | null;
  combined_plate: string | null;
}

const videos = ref<Record<string, string>>({});
const tracks = ref<Record<string, Record<string, TrackInfo>>>({});
const groups = ref<Group[]>([]);
const uncertain = ref<Group[]>([]);
const session = ref("");
const loadError = ref("");
const busy = ref(false);
const finalizeError = ref("");

// Decision state: confident groups start ACCEPTED, uncertain start REJECTED.
const acceptedConfident = ref<boolean[]>([]);
const acceptedUncertain = ref<boolean[]>([]);

onMounted(async () => {
  try {
    const raw = await invoke<string>("read_match_review", {
      path: props.reviewPath,
    });
    const data = JSON.parse(raw);
    videos.value = data.videos || {};
    tracks.value = data.tracks || {};
    groups.value = data.groups || [];
    uncertain.value = data.uncertain || [];
    session.value = data.session || "";
    acceptedConfident.value = groups.value.map(() => true);
    acceptedUncertain.value = uncertain.value.map(() => false);
  } catch (e) {
    loadError.value = String(e);
  }
});

function trackOf(m: [string, number]): TrackInfo | null {
  return tracks.value[m[0]]?.[String(m[1])] ?? null;
}

function fmtTime(s: number): string {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

async function finalize() {
  busy.value = true;
  finalizeError.value = "";
  try {
    const chosen = [
      ...groups.value.filter((_, i) => acceptedConfident.value[i]),
      ...uncertain.value.filter((_, i) => acceptedUncertain.value[i]),
    ].map((g) => g.members);
    const report = await invoke<string>("finalize_match", {
      session: session.value,
      decisions: JSON.stringify({ groups: chosen }),
    });
    emit("done", report);
  } catch (e) {
    finalizeError.value = String(e);
  } finally {
    busy.value = false;
  }
}
</script>

<template>
  <div class="review card card-violet">
    <h3>Έλεγχος συσχετίσεων μεταξύ βίντεο</h3>
    <p class="hint">
      Ο αλγόριθμος προτείνει — εσείς αποφασίζετε. Απορρίψτε λανθασμένες
      «σίγουρες» συσχετίσεις και εγκρίνετε όσες πιθανές ισχύουν· μετά θα
      δημιουργηθεί μία συνδυαστική αναφορά για όλα τα βίντεο.
    </p>

    <div v-if="loadError" class="error">{{ loadError }}</div>

    <template v-else>
      <h4>
        Σίγουρες συσχετίσεις ({{ groups.length }})
        <span class="sub">— προεπιλεγμένα ΝΑΙ· ξετσεκάρετε ό,τι είναι λάθος</span>
      </h4>
      <p v-if="!groups.length" class="empty">Καμία αυτόματη συσχέτιση.</p>
      <div
        v-for="(g, i) in groups"
        :key="'c' + i"
        class="grouprow"
        :class="{ off: !acceptedConfident[i] }"
      >
        <label class="acc">
          <input type="checkbox" v-model="acceptedConfident[i]" />
          <span>Συσχέτιση</span>
        </label>
        <div class="members">
          <div v-for="m in g.members" :key="m[0] + m[1]" class="member">
            <img
              v-if="trackOf(m)?.thumb"
              :src="'data:image/jpeg;base64,' + trackOf(m)!.thumb"
            />
            <div class="meta">
              <strong>{{ trackOf(m)?.class?.toUpperCase() }} #{{ m[1] }}</strong>
              <span class="vid">{{ m[0] }}</span>
              <span v-if="trackOf(m)"
                >{{ fmtTime(trackOf(m)!.first_seen) }} –
                {{ fmtTime(trackOf(m)!.last_seen) }}</span
              >
              <span v-if="trackOf(m)?.plate" class="plate">🚘 {{ trackOf(m)!.plate }}</span>
            </div>
          </div>
        </div>
        <div class="evidence">
          <span class="badge" :class="{ weak: g.evidence === 'appearance' }">{{
            g.evidence === "appearance"
              ? "ΜΟΝΟ ΕΜΦΑΝΙΣΗ — χαμηλή βεβαιότητα"
              : g.evidence?.includes("plate")
                ? "ΠΙΝΑΚΙΔΑ" + (g.evidence.includes("appearance") ? " + ΕΜΦΑΝΙΣΗ" : "")
                : g.evidence
          }}</span>
          <span v-if="g.combined_plate" class="plate">Συνδυαστική: {{ g.combined_plate }}</span>
        </div>
      </div>

      <h4>
        Πιθανές συσχετίσεις ({{ uncertain.length }})
        <span class="sub">— προεπιλεγμένα ΟΧΙ· τσεκάρετε όσες ισχύουν</span>
      </h4>
      <p v-if="!uncertain.length" class="empty">Καμία πιθανή συσχέτιση προς έλεγχο.</p>
      <div
        v-for="(g, i) in uncertain"
        :key="'u' + i"
        class="grouprow uncertain"
        :class="{ off: !acceptedUncertain[i] }"
      >
        <label class="acc">
          <input type="checkbox" v-model="acceptedUncertain[i]" />
          <span>Συσχέτιση</span>
        </label>
        <div class="members">
          <div v-for="m in g.members" :key="m[0] + m[1]" class="member">
            <img
              v-if="trackOf(m)?.thumb"
              :src="'data:image/jpeg;base64,' + trackOf(m)!.thumb"
            />
            <div class="meta">
              <strong>{{ trackOf(m)?.class?.toUpperCase() }} #{{ m[1] }}</strong>
              <span class="vid">{{ m[0] }}</span>
              <span v-if="trackOf(m)"
                >{{ fmtTime(trackOf(m)!.first_seen) }} –
                {{ fmtTime(trackOf(m)!.last_seen) }}</span
              >
              <span v-if="trackOf(m)?.plate" class="plate">🚘 {{ trackOf(m)!.plate }}</span>
            </div>
          </div>
        </div>
        <div class="evidence">
          <span class="badge weak">{{ g.evidence }}</span>
        </div>
      </div>

      <div v-if="finalizeError" class="error">{{ finalizeError }}</div>

      <div class="actions">
        <button class="primary" :disabled="busy" @click="finalize">
          {{ busy ? "Δημιουργία..." : "Δημιουργία συνδυαστικής αναφοράς" }}
        </button>
        <button class="secondary" :disabled="busy" @click="emit('skip')">
          Παράλειψη
        </button>
      </div>
    </template>
  </div>
</template>

<style scoped>
.review {
  overflow-y: auto;
  max-height: 100%;
}
.hint {
  color: var(--text-secondary);
  font-size: 0.88rem;
  margin: 6px 0 14px;
}
h4 {
  margin: 18px 0 8px;
}
h4 .sub {
  font-weight: 400;
  font-size: 0.8rem;
  color: var(--text-secondary);
}
.empty {
  color: var(--text-secondary);
  font-size: 0.85rem;
}
.grouprow {
  display: flex;
  align-items: center;
  gap: 14px;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 10px 12px;
  margin-bottom: 8px;
  background: var(--bg-card);
  transition: opacity 0.15s ease;
}
.grouprow.off {
  opacity: 0.45;
}
.grouprow.uncertain {
  border-style: dashed;
}
.acc {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  font-size: 0.75rem;
  color: var(--text-secondary);
  min-width: 74px;
  cursor: pointer;
}
.acc input {
  width: 18px;
  height: 18px;
  accent-color: var(--accent);
  cursor: pointer;
}
.members {
  display: flex;
  gap: 12px;
  flex: 1;
  flex-wrap: wrap;
}
.member {
  display: flex;
  gap: 8px;
  align-items: center;
}
.member img {
  width: 72px;
  height: 54px;
  object-fit: cover;
  border-radius: 6px;
  border: 1px solid var(--border);
}
.meta {
  display: flex;
  flex-direction: column;
  font-size: 0.78rem;
  color: var(--text-secondary);
}
.meta strong {
  color: var(--text-primary);
  font-size: 0.82rem;
}
.vid {
  max-width: 180px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.plate {
  color: var(--accent);
  font-weight: 600;
}
.evidence {
  display: flex;
  flex-direction: column;
  gap: 4px;
  align-items: flex-end;
  min-width: 150px;
}
.badge {
  font-size: 0.7rem;
  font-weight: 700;
  padding: 3px 8px;
  border-radius: 6px;
  background: rgba(15, 157, 110, 0.15);
  color: var(--success);
  border: 1px solid var(--success);
  text-align: center;
}
.badge.weak {
  background: rgba(217, 164, 43, 0.12);
  color: #b07d10;
  border-color: #b07d10;
}
.error {
  color: var(--danger);
  font-size: 0.85rem;
  margin: 10px 0;
}
.actions {
  display: flex;
  gap: 10px;
  margin-top: 16px;
}
</style>
