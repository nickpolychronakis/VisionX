<script setup lang="ts">
// Match-review screen (user workflow): after a multi-camera analysis, the
// investigator SEES what was auto-matched (and can reject it — the
// algorithm does err) and what is merely suspected (and can accept it).
// A MANUAL PAIRING BOARD (user request, dashcam front/back case) covers
// everything the suggestions miss: pick a track in each column, link them.
// The decisions then regenerate ONE combined report via vision.py
// --finalize-match. Human judgment always outranks the algorithm.
import { computed, onMounted, ref } from "vue";
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
  // Ready-made Greek label + tier from python (cross_match.py is the ONE
  // translation table — the Vue side renders verbatim, never translates).
  evidence_label: string | null;
  evidence_tier: "strong" | "weak" | null;
  score: number | null;
  combined_plate: string | null;
}

type Member = [string, number];

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

// Manual pairing board state: one selected track per column + linked pairs.
const videoNames = computed(() => Object.keys(videos.value));
const leftVideo = ref("");
const rightVideo = ref("");
const leftSel = ref<Member | null>(null);
const rightSel = ref<Member | null>(null);
const manualPairs = ref<[Member, Member][]>([]);

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
    // Board defaults: first two videos side by side; a single-video session
    // (same-video re-appearances) pairs within the one video.
    const names = videoNames.value;
    leftVideo.value = names[0] ?? "";
    rightVideo.value = names[1] ?? names[0] ?? "";
  } catch (e) {
    loadError.value = String(e);
  }
});

function trackOf(m: Member): TrackInfo | null {
  return tracks.value[m[0]]?.[String(m[1])] ?? null;
}

function fmtTime(s: number): string {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

const memberKey = (m: Member) => `${m[0]}|${m[1]}`;
const sameMember = (a: Member | null, b: Member) =>
  !!a && a[0] === b[0] && a[1] === b[1];

// Tracks already inside an ACCEPTED suggestion or a manual pair — shown
// dimmed with a ✓ so the investigator sees remaining work at a glance.
const pairedKeys = computed(() => {
  const keys = new Set<string>();
  groups.value.forEach((g, i) => {
    if (acceptedConfident.value[i]) g.members.forEach((m) => keys.add(memberKey(m)));
  });
  uncertain.value.forEach((g, i) => {
    if (acceptedUncertain.value[i]) g.members.forEach((m) => keys.add(memberKey(m)));
  });
  manualPairs.value.forEach(([a, b]) => {
    keys.add(memberKey(a));
    keys.add(memberKey(b));
  });
  return keys;
});

function columnTracks(video: string, sortNear: Member | null): Member[] {
  const entries = Object.keys(tracks.value[video] ?? {}).map(
    (tid) => [video, Number(tid)] as Member,
  );
  const ref_t = sortNear ? trackOf(sortNear) : null;
  return entries.sort((a, b) => {
    const ta = trackOf(a)!;
    const tb = trackOf(b)!;
    if (ref_t) {
      // TIME PROXIMITY sort (front/back dashcams record the same moment:
      // the matching track sits at nearly the same timestamp — it floats
      // to the top the instant a left-column track is selected).
      return (
        Math.abs(ta.first_seen - ref_t.first_seen) -
        Math.abs(tb.first_seen - ref_t.first_seen)
      );
    }
    return ta.first_seen - tb.first_seen;
  });
}

const leftTracks = computed(() => columnTracks(leftVideo.value, null));
const rightTracks = computed(() => columnTracks(rightVideo.value, leftSel.value));

function pick(side: "left" | "right", m: Member) {
  const sel = side === "left" ? leftSel : rightSel;
  sel.value = sameMember(sel.value, m) ? null : m;
}

function linkPair() {
  if (!leftSel.value || !rightSel.value) return;
  if (sameMember(leftSel.value, rightSel.value)) return; // same track twice
  manualPairs.value.push([leftSel.value, rightSel.value]);
  leftSel.value = null;
  rightSel.value = null;
}

function unlinkPair(i: number) {
  manualPairs.value.splice(i, 1);
}

async function finalize() {
  busy.value = true;
  finalizeError.value = "";
  try {
    // Accepted suggestions + manual pairs, all as member groups — the
    // python finalize coalesces chains (A-B + B-C → one object).
    const chosen: Member[][] = [
      ...groups.value.filter((_, i) => acceptedConfident.value[i]).map((g) => g.members),
      ...uncertain.value.filter((_, i) => acceptedUncertain.value[i]).map((g) => g.members),
      ...manualPairs.value.map(([a, b]) => [a, b]),
    ];
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
    <h3>Έλεγχος συσχετίσεων</h3>
    <p class="hint">
      Ο αλγόριθμος προτείνει — εσείς αποφασίζετε. Ελέγξτε τις προτάσεις και
      συνδέστε χειροκίνητα ό,τι λείπει· μετά θα δημιουργηθεί μία συνδυαστική
      αναφορά για όλα τα βίντεο.
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
          <div v-for="m in g.members" :key="memberKey(m)" class="member">
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
          <span class="badge" :class="{ weak: g.evidence_tier === 'weak' }">{{
            g.evidence_label ?? g.evidence
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
          <div v-for="m in g.members" :key="memberKey(m)" class="member">
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
          <span class="badge weak">{{ g.evidence_label ?? g.evidence }}</span>
        </div>
      </div>

      <h4>
        Χειροκίνητη αντιστοίχιση
        <span class="sub"
          >— επιλέξτε ένα όχημα σε κάθε στήλη και πατήστε «Σύνδεση»· η δεξιά
          στήλη ταξινομείται κατά χρονική εγγύτητα με την επιλογή σας
          (μπρος/πίσω κάμερες: το ίδιο όχημα εμφανίζεται σχεδόν ταυτόχρονα)</span
        >
      </h4>
      <div class="board">
        <div class="col">
          <select v-model="leftVideo" class="vidsel">
            <option v-for="v in videoNames" :key="'l' + v" :value="v">{{ v }}</option>
          </select>
          <div class="tracklist">
            <div
              v-for="m in leftTracks"
              :key="memberKey(m)"
              class="trackrow"
              :class="{
                selected: sameMember(leftSel, m),
                paired: pairedKeys.has(memberKey(m)),
              }"
              @click="pick('left', m)"
            >
              <img
                v-if="trackOf(m)?.thumb"
                :src="'data:image/jpeg;base64,' + trackOf(m)!.thumb"
              />
              <div class="meta">
                <strong>{{ trackOf(m)?.class?.toUpperCase() }} #{{ m[1] }}</strong>
                <span>{{ fmtTime(trackOf(m)!.first_seen) }}</span>
                <span v-if="trackOf(m)?.plate" class="plate">{{ trackOf(m)!.plate }}</span>
              </div>
              <span v-if="pairedKeys.has(memberKey(m))" class="check">✓</span>
            </div>
          </div>
        </div>

        <div class="linkcol">
          <button
            class="primary linkbtn"
            :disabled="!leftSel || !rightSel"
            @click="linkPair"
            title="Σύνδεση των δύο επιλεγμένων ως ΙΔΙΟ όχημα"
          >
            Σύνδεση ▶◀
          </button>
          <div class="pairs">
            <div v-for="(p, i) in manualPairs" :key="'p' + i" class="pairchip">
              <span
                >{{ trackOf(p[0])?.class?.toUpperCase() }} #{{ p[0][1] }} ↔
                #{{ p[1][1] }}</span
              >
              <button class="x" @click="unlinkPair(i)" title="Αποσύνδεση">×</button>
            </div>
          </div>
        </div>

        <div class="col">
          <select v-model="rightVideo" class="vidsel">
            <option v-for="v in videoNames" :key="'r' + v" :value="v">{{ v }}</option>
          </select>
          <div class="tracklist">
            <div
              v-for="m in rightTracks"
              :key="memberKey(m)"
              class="trackrow"
              :class="{
                selected: sameMember(rightSel, m),
                paired: pairedKeys.has(memberKey(m)),
              }"
              @click="pick('right', m)"
            >
              <img
                v-if="trackOf(m)?.thumb"
                :src="'data:image/jpeg;base64,' + trackOf(m)!.thumb"
              />
              <div class="meta">
                <strong>{{ trackOf(m)?.class?.toUpperCase() }} #{{ m[1] }}</strong>
                <span>{{ fmtTime(trackOf(m)!.first_seen) }}</span>
                <span v-if="trackOf(m)?.plate" class="plate">{{ trackOf(m)!.plate }}</span>
              </div>
              <span v-if="pairedKeys.has(memberKey(m))" class="check">✓</span>
            </div>
          </div>
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
  min-height: 0;
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

/* Manual pairing board */
.board {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  gap: 12px;
  align-items: start;
}
.vidsel {
  width: 100%;
  margin-bottom: 8px;
}
.tracklist {
  max-height: 340px;
  overflow-y: auto;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 6px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.trackrow {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 8px;
  border: 1px solid transparent;
  border-radius: 8px;
  cursor: pointer;
  position: relative;
}
.trackrow:hover {
  background: var(--bg-card);
}
.trackrow.selected {
  border-color: var(--accent);
  background: rgba(79, 140, 255, 0.12);
}
.trackrow.paired {
  opacity: 0.55;
}
.trackrow img {
  width: 64px;
  height: 48px;
  object-fit: cover;
  border-radius: 6px;
  border: 1px solid var(--border);
}
.trackrow .check {
  margin-left: auto;
  color: var(--success);
  font-weight: 700;
}
.linkcol {
  display: flex;
  flex-direction: column;
  gap: 10px;
  align-items: center;
  padding-top: 40px;
  min-width: 170px;
}
.linkbtn {
  white-space: nowrap;
}
.pairs {
  display: flex;
  flex-direction: column;
  gap: 6px;
  width: 100%;
}
.pairchip {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  font-size: 0.78rem;
  border: 1px solid var(--accent);
  border-radius: 8px;
  padding: 4px 8px;
  color: var(--text-primary);
  background: rgba(79, 140, 255, 0.1);
}
.pairchip .x {
  background: transparent;
  color: var(--danger);
  padding: 0 4px;
  font-size: 1rem;
}
</style>
