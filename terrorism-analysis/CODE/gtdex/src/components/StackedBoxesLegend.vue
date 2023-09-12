<template>
  <g class="stacked-boxes-legend">
    <rect :x="x"
          :y="y"
          :height="height"
          :width="width + 10"
          rx="15"
          stroke="white"
          fill="white"
          class="stacked-boxes-legend container"
    ></rect>
    <text :x="x"
          :y="y - 10"
          class="stacked-boxes-legend title"
    >{{ title }}</text>
    <g v-for="key in keys"
       class="stacked-boxes-legend keys">
      <rect :x="key.x"
            :y="key.y"
            :stroke="key.color"
            :fill="key.color"
            :height="key.height"
            :width="key.height"
      ></rect>
      <text class="stacked-boxes-legend key label"
            :x="key.x + key.height + 8"
            :y="key.y + (key.height / 2) + 8"
            alignment-baseline="bottom"
      >{{key.label}}
      </text>
    </g>
  </g>
</template>
<script>
export default {
  name: 'StackedBoxesLegend',
  props: {
    x: {
      type: Number,
      default: () => 0,
    },
    y: {
      type: Number,
      default: () => 0,
    },
    height: {
      type: Number,
      default: () => 500,
    },
    width: {
      type: Number,
      default: () => 100,
    },
    title: {
      type: String,
      default: () => '',
    },
    min: {
      type: Number,
      default: () => 0,
    },
    max: {
      type: Number,
      default: () => 100
    },
    exponent: {
      type: Number,
      default: () => 1,
    },
  },
  computed: {
    keys() {
      const that = this;
      const colorScale = d3.scalePow()
        .exponent(1)
        .domain([this.min, this.max])
        .range([0, 1])
        .nice()

      const keys = colorScale.ticks();
      const keyHeight = (this.height - 30) / keys.length
      const offsetX = that.x + 15
      const offsetY = that.y + 15
      return keys.map((k, i) => {
        const label = k.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",")
        return {
          x: offsetX,
          y: offsetY + ((keys.length-1) * keyHeight) - (i * keyHeight),
          height: keyHeight - 5,
          color: d3.interpolateReds(colorScale(k)),
          label,
        }
      });

    },
  },
}
</script>
<style>
.stacked-boxes-legend.container {
  opacity: 75%;
}
.stacked-boxes-legend.title {
  font-weight: bold;
  font-size: 1em;
}

.stacked-boxes-legend.label {
  font-size: 1.1em;
  font-weight: bold;
}
</style>
