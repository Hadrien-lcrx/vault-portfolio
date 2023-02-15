<template>
  <g class="choropleth">
    <path v-for="feature in features"
          :key="feature.id"
          :id="feature.id"
          :data-name="feature.properties.name"
          fill="gray"
          class="path geometry"
          stroke-width=".5px"
          stroke="white"
    >
    </path>
  </g>
</template>

<script>
export const ChoroplethMapProperties = {
  x: {
    type: Number,
    default: () => 0,
  },
  y: {
    type: Number,
    default: () => 0,
  },
  canvasHeight: {
    type: Number,
    default: () => 0,
  },
  canvasWidth: {
    type: Number,
    default: () => 0,
  },
  features: {
    type: Array,
    default: () => [],
  },
};
export const ChoroplethMapEvents = {
  CLEAR: 'clear',
  CREATED: 'created',
};

export default {
  name: 'ChoroplethMap',
  props: ChoroplethMapProperties,
  computed: {
    projection() {
      return d3.geoMercator()
        .translate([
          (this.canvasWidth / 2) + this.x,
          (this.canvasHeight / 2) + this.y])
        .scale(this.canvasWidth / (2 * Math.PI));
    },
    path() {
      return d3.geoPath().projection(this.projection);
    },
  },
  methods: {
    draw() {
      const geometries = d3.selectAll('.path.geometry')
        .data(this.features)
        .attr('d', this.path)
        .on('click', (e) => {
          this.$emit('clicked', e);
        });
      this.$emit(ChoroplethMapEvents.CREATED, { geometries });
    },
  },
};
</script>
<style>
.path.geometry {
  cursor: pointer;
}
</style>
