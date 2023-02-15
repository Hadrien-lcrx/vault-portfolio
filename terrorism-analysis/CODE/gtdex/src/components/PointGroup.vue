<template>
  <g class="point-group" :id="groupName">
    <circle v-for="(circle, i) in circles"
            :key="i"
            class="point grow"
            v-bind="circle"
            @mouseover="onmouseover"
            @mouseout="onmouseout"
    ></circle>
  </g>
</template>
<script>
export default {
  name: 'PointGroup',
  props: {
    groupName: {
      type: String,
      default: () => '',
    },
    canvasWidth: {
      type: Number,
      default: () => 0,
    },
    canvasHeight: {
      type: Number,
      default: () => 0,
    },
    margin: {
      type: Object,
      default: () => ({top: 0, bottom: 0, left: 0, right: 0})
    },
    points: {
      type: Array,
      default: () => [],
    }
  },
  computed: {
    circles() {
      const projection = d3.geoMercator();
      projection.scale(this.canvasWidth / (2 * Math.PI))
        .translate([
          (this.canvasWidth + this.margin.left + this.margin.right) / 2,
          (this.canvasHeight + this.margin.top + this.margin.bottom) / 2,
        ]);

      return this.points.map(p => {
        const [cx, cy] = projection([p['data-longitude'], p['data-latitude']])
        return {...p, cx, cy}
      })
    },
  },
  methods: {
    onmouseover(e) {
      const point = d3.select(e.target);
      point
        .attr('r', point.attr('r') * 2)
      this.$emit('mouseoverPoint', e.target);
      e.stopPropagation();
    },
    onmouseout(e) {
      const point = d3.select(e.target);
      point.attr('r', point.attr('r') / 2)
      this.$emit('mouseoutPoint', e.target);
    },
  },
}
</script>
<style lang="scss">
</style>
