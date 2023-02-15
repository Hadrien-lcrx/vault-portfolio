<template>
  <div class="incidentcard mdc-card"
       :style="`height: ${height}; width: ${width}; minWidth: 400px;`"
  >
    <div class="incidentcard header">
      <div class="incidentcard header section" style="align-items: flex-start">
        <span class="incidentcard title">{{ incident.city }}, {{ incident.country_txt }}</span>
        <span class="incidentcard subtitle">{{ incidentDate }}</span>
      </div>
      <div class="incidentcard header section" style="align-items: flex-end; margin: 10px 10px;">
        <span class="dismiss" @click="dismiss">Dismiss</span>
      </div>
    </div>

    <hr class="incidentcard divider">
    <div class="incidentcard section body">
      <div class="incidentcard data-group">
        <div class="incidentcard data">
          <span class="label">Perpetrator</span>
          <span class="value">{{ incident.gname ? incident.gname : 'Unknown' }}</span>
        </div>
        <div class="incidentcard data">
          <span class="label">Target</span>
          <span class="value">{{ incident.target1 ? incident.target1 : 'Unknown' }}</span>
        </div>
      </div>
      <div class="incidentcard data-group">
        <div class="incidentcard data">
          <span class="label">Wounded</span>
          <span class="value">{{ incident.nwound ? incident.nwound : 'None' }}</span>
        </div>
        <div class="incidentcard data">
          <span class="label">Suicide Attack</span>
          <span class="value">
            {{ incident.suicide === '1' ? 'Yes' : 'No' }}
          </span>
        </div>
      </div>
      <div class="incidentcard data-group">
        <div class="incidentcard data">
          <span class="label">Successful Attack</span>
          <span class="value">
            {{ incident.success === '1' ? 'Yes' : 'No' }}
          </span>
        </div>
        <div class="incidentcard data">
          <span class="label">Weapon/Attack Method</span>
          <span class="value">
            {{ incident.weaptype1_txt ? incident.weaptype1_txt : 'Unknown' }}
          </span>
        </div>
      </div>
      <div class="incidentcard data-group">
        <div class="incidentcard data">
          <span class="label">Sub-Type</span>
          <span class="value">
            {{ incident.weapsubtype1_txt ? incident.weapsubtype1_txt : 'Unknown' }}
          </span>
        </div>
        <div class="incidentcard data">
          <span class="label"></span>
        </div>
      </div>
      <div class="incidentcard data-group">
        <div class="incidentcard data">
          <span class="label">Doubt Terrorism</span>
          <span class="value">
            {{
              incident.doubtterr === '-9' ? 'Unknown'
                : incident.doubtterr === '1' ? 'Yes'
                : 'No'
            }}
          </span>
        </div>
        <div class="incidentcard data">
          <span class="label">Has Hostages</span>
          <span class="value">
            {{
              incident.ishostkid === '-9' ? 'Unknown'
                : incident.ishostkid === '1' ? 'Yes' : 'No'
            }}
          </span>
        </div>
      </div>
      <div class="incidentcard data-group">
        <div class="incidentcard data">
          <span class="label">Casualties</span>
          <span class="value" style="font-size: 1.5em; font-weight: bold">
          {{ incident.nkill ? formatNumber(incident.nkill) : 'None' }}
        </span>
        </div>
        <div class="incidentcard data">
          <span class="label" style="display: inline-flex; flex-direction: row; align-items: center">
            <i v-if="!tip"
               class="fas fa-info-circle"
               style="margin-right: 2px; color: dodgerblue; cursor: pointer"
               @click="showTip"
            >
            </i>
            <i v-if="tip"
               class="fas fa-times-circle"
               style="margin-right: 2px; color: dodgerblue; cursor: pointer"
               @click="showTip"
            ></i>
            Predicted Casualties
          </span>
          <span class="value" style="font-size: 1.5em; font-weight: bold">
            {{ predictions[incident.eventid] ? predictions[incident.eventid] : 'N/A' }}
          </span>
        </div>
      </div>
    </div>

    <hr class="incidentcard divider" v-if="tip">
    <div class="incidentcard section footer" v-if="tip">
      <div>
        <p>
          Predicted values are derived from an ElasticNet Linear Regressor. Of the 135 columns in
          the original Global Terrorism Database, 7 features were identified as the best indicators
          for estimating the number of casualties in a given incident.
        </p>
        <p>
          Incidents with a value of "N/A" were used in the training set and excluded from the test
          set.
        </p>
        <span style="font-weight: bold; font-size: 1.2em">Model Features</span>
        <ol>
          <li>Number of People wounded from the incident</li>
          <li>Whether or not the attack was a suicide attack</li>
          <li>Whether or not the attack was successful</li>
          <li>The primary attack weapon/methodology</li>
          <li>The primary subcategory of the attack weapon/methodology</li>
          <li>Whether or not hostages were involved in the situation</li>
          <li>Whether or not the incident met all criteria to be widely considered terrorism</li>
        </ol>
        <p>
          See the
          <a href="https://www.start.umd.edu/gtd/downloads/Codebook.pdf">
            Global Terrorism Database Codebook
          </a> for details about each feature.
        </p>
      </div>
    </div>
    <div class="spacer"></div>
  </div>
</template>
<script>
import predictions from '@/assets/pred.json';
const months = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December"
]
export default {
  name: 'IncidentCard',
  props: {
    x: {
      type: Number,
      default: () => 15,
    },
    y: {
      type: Number,
      default: () => 15,
    },
    height: {
      type: String,
      default: () => 'auto',
    },
    width: {
      type: String,
      default: () => '400px',
    },
    incident: {
      type: Object,
      default: () => ({
        city: 'New York City',
        country_txt: 'United States of America',
        iyear: 2001,
        imonth: 9,
        iday: 11,
      }),
    },
  },
  computed: {
    incidentDate() {
      const year = this.incident.iyear.toString();
      const month = months[this.incident.imonth - 1];
      const date = this.incident.iday.toString().padStart(2, '0');
      return `${month} ${date}, ${year}`
    },
  },
  data() {
    return {
      tip: false,
      predictions,
    }
  },
  methods: {
    dismiss() {
      this.$emit('dismiss');
    },
    formatNumber(f) {
      return f.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",")
    }, // Formats a number for commas
    showTip() {
      this.tip = !this.tip;
    },
  },
};
</script>
<style lang="scss">
.incidentcard {
  text-align: left;
  padding: 0 5px;
}

.incidentcard.header {
  display: inline-flex;
  flex-direction: row;
  padding-top: 2px;
  width: 100%;
}

.incidentcard.header.section {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}

.incidentcard.title {
  font-weight: bolder;
  font-size: 1.5em;
}

.incidentcard.subtitle {
  font-size: 1em;
  font-weight: bold;
}

.dismiss {
  font-weight: bolder;
  color: #333;
  cursor: pointer;
}

.incidentcard.divider {
  border-top: 0.25px solid #ddd;
  margin: 5px 0;
}

.incidentcard.section {
  display: inline-flex;
  flex-direction: row;
  justify-content: space-between;
  align-items: flex-start;
}

.incidentcard.data {
  display: inline-flex;
  flex-direction: column;
  align-items: stretch;
  margin: 5px 0;
}

.incidentcard .label {
  font-size: 0.8em;
  font-weight: bolder;
  margin: 2.5px 0;
  color: #bbb;
}
</style>
