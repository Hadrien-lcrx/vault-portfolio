<template>
  <CardControl :classs="'querycard'"
               :width="width"
               :height="height">
    <form style="width: 100%" class="querycard form">
      <div class="query-title margin-bottom">
        <label class="mdc-text-field mdc-text-field--filled title square-edge">
          <span class="mdc-text-field__ripple"></span>
          <input type="text"
                 class="mdc-text-field__input"
                 aria-labelledby="title-label"
                 name="title"
                 v-model="title"
          >
          <span class="mdc-floating-label title-label">Title</span>
          <span class="mdc-line-ripple"></span>
        </label>
      </div>

      <div class="query-country margin-bottom">
        <label style="display: flex; flex-direction: column; justify-content: space-between">
          Country
          <select v-model="country">
            <option v-for="(country, i) in countryNames" :key="i">{{ country }}</option>
          </select>
        </label>
      </div>

      <div class="query-years margin-bottom">
        <label>Incident Year</label>
        <div class="year-slider mdc-slider mdc-slider--range mdc-slider--discrete" data-step="1">
          <div class="mdc-slider__track">
            <div class="mdc-slider__track--inactive"></div>
            <div class="mdc-slider__track--active">
              <div class="mdc-slider__track--active_fill"></div>
            </div>
          </div>
          <div class="mdc-slider__thumb"
               role="slider" tabindex="0"
               aria-label="Year"
               :aria-valuemin="minYear"
               :aria-valuemax="maxYear"
               :aria-valuenow="minYear">
            <div class="mdc-slider__value-indicator-container">
              <div class="mdc-slider__value-indicator">
        <span :class="`mdc-slider__value-indicator-text startYear-${name}`">
          {{ minYear }}
        </span>
              </div>
            </div>
            <div class="mdc-slider__thumb-knob"></div>
          </div>
          <div class="mdc-slider__thumb"
               role="slider"
               tabindex="0"
               aria-label="Year"
               :aria-valuemin="minYear"
               :aria-valuemax="maxYear"
               :aria-valuenow="maxYear">
            <div class="mdc-slider__value-indicator-container">
              <div class="mdc-slider__value-indicator">
        <span :class="`mdc-slider__value-indicator-text endYear-${name}`">
          {{ maxYear }}
        </span>
              </div>
            </div>
            <div class="mdc-slider__thumb-knob"></div>
          </div>
        </div>
        <div class="year-labels">
          <label>{{ minYear }}</label><label>{{ maxYear }}</label>
        </div>
      </div>

      <div class="query-fatal-only margin-bottom">
        <label :for="`fatal-switch-${name}`" class="margin-right">Show Non-Fatal</label>
        <div class="fatal-switch mdc-switch">
          <div class="mdc-switch__track"></div>
          <div class="mdc-switch__thumb-underlay">
            <div class="mdc-switch__thumb"></div>
            <input type="checkbox" :id="`fatal-switch-${name}`"
                   class="mdc-switch__native-control"
                   role="switch"
                   aria-checked="false"
                   v-model="showNonFatal"
            >
          </div>
        </div>
      </div>

      <div class="query-submit margin-bottom">
        <div class="mdc-touch-target-wrapper">
          <button class="query-btn mdc-button mdc-button--raised mdc-button--touch" @click="submit">
            <div class="mdc-button__ripple"></div>
            <span class="mdc-button__label">Query</span>
            <div class="mdc-button__touch"></div>
          </button>
        </div>
      </div>
    </form>
  </CardControl>
</template>
<script>
import worldCountries from '@/assets/world_countries.json';
import CardControl from "@/components/CardControl";
import {MDCTextField} from "@material/textfield/component";
import {MDCRipple} from "@material/ripple/component";
import {MDCSlider} from "@material/slider/component";
import {MDCSwitch} from "@material/switch/component";

export default {
  name: 'QueryCard',
  components: {CardControl},
  props: {
    name: {
      type: String,
      default: () => ''
    },
    width: {
      type: String,
      default: () => '100%'
    },
    height: {
      type: String,
      default: () => '300px'
    },
    minYear: {
      type: Number,
      default: () => 1970,
    },
    maxYear: {
      type: Number,
      default: () => 2018,
    },
  },
  computed: {
    countryNames() {
      return ['All', ...worldCountries.features.reduce((acc, c) => {
        acc.push(c.properties.name)
        return acc;
      }, []).sort()]
    },
  },
  data() {
    return {
      title: 'Query 1',
      country: 'All',
      startYear: 1970,
      endYear: 1997,
      showNonFatal: false,
    }
  },
  mounted() {
    const title = new MDCTextField(document.querySelector('.title'))
    const years = new MDCSlider(document.querySelector('.year-slider'));
    const onlyFatal = new MDCSwitch(document.querySelector('.fatal-switch'))
    const query = new MDCRipple(document.querySelector('.query-btn'))
  },
  methods: {
    submit(e) {
      e.preventDefault();
      this.startYear = parseInt(d3.select(`.startYear-${this.name}`).text())
      this.endYear = parseInt(d3.select(`.endYear-${this.name}`).text())
      this.$emit('querySubmit', e, this);
    }
  },
}
</script>
<style lang="scss">
@use "~@material/floating-label/mdc-floating-label";
@use "~@material/line-ripple/mdc-line-ripple";
@use "~@material/notched-outline/mdc-notched-outline";
@use "~@material/textfield";
@use "~@material/button";
@use "~@material/card";
@use "~@material/slider/styles";
@use "~@material/switch";

@include button.core-styles;
@include card.core-styles;
@include switch.core-styles;
@include textfield.core-styles;

.margin-bottom {
  margin-bottom: 15px;
}

.margin-right {
  margin-right: 15px;
}

.mdc-text-field {
  display: flex;
}

.year-labels {
  display: flex;
  justify-content: space-between;
  padding: 0 16px
}

.query-fatal-only {
  display: flex;
  justify-content: flex-start;
  padding: 0 16px;
}

.query-btn {
  display: block;
  margin: auto;
}

.query-btn button {
  border: 1px solid #999;
}

.form-title, .form-title h4 {
  background: #14342B;
  color: white;
  font-size: 2em;
  font-weight: bolder;
  padding: 8px 0;
}
</style>
