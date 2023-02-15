<template>
  <div>
    <h1>{{ formatNumber(totalCasualties) }} Casualties from Terrorism</h1>
    <h4>({{ minYear }} to {{ maxYear }})</h4>
    <div id="gtd-content">
      <div class="gtd-row incident-container">
        <!--        <div class="incident-placeholder" v-if="!focusIncidentId.length">-->
        <!--          <em>Query for incidents below and then mouse over a point to see details.</em>-->
        <!--        </div>-->
      </div>
      <div class="gtd-row">
        <div id="left-panel" class="gtd-item">
          <QueryCard class="margin-bottom-small"
                     name="querycard"
                     :min-year="minYear"
                     :max-year="maxYear"
                     @querySubmit="querySubmit"></QueryCard>
          <ResultCard v-for="(query, i) in queries"
                      :key="i"
                      v-bind="query"></ResultCard>
        </div>
        <div id="map-panel" class="gtd-item">
          <svg
            :height="canvasHeight"
            :width="canvasWidth"
            class="gtd choropleth canvas"
            id="gtd-canvas"
            @click="reset"
          >
            <ChoroplethMap id="gtd-map"
                           :features="features"
                           :canvas-width="width"
                           :canvas-height="height"
                           :x="margin.left"
                           :y="margin.top"
                           @created="handleCreated"
                           @clicked="handleClicked"
                           ref="gtdMap"
            >
            </ChoroplethMap>
            <PointGroup v-for="(group, i) in pointGroups"
                        v-bind="group"
                        :key="i"
                        :canvas-width="width"
                        :canvas-height="height"
                        :margin="margin"
                        @mouseoverPoint="mouseoverPoint"
            ></PointGroup>
            <StackedBoxesLegend v-if="!focused"
                                :x="margin.left"
                                :y="(canvasHeight / 2) + margin.top"
                                :height="canvasHeight / 2.5"
                                :width="100"
                                :exponent="exponent"
                                :min="0"
                                :max="mostCasualties"
                                title="Total Casualties"
            ></StackedBoxesLegend>
          </svg>
          <transition name="slide-up">
            <IncidentCard id="gtd-incident"
                          v-if="focusIncidentId.length"
                          :incident="incidents[focusIncidentId]"
                          width="890px"
                          @dismiss="dismiss"
            ></IncidentCard>
          </transition>
        </div>
      </div>
    </div>
    <div id="max-results-text">Showing top {{maxResults}} incidents per query</div>
    <div class="spacer"></div>
  </div>
</template>
<script>
import {GtdAPIClient as gtd} from '@/api/GTDClient';
import ChoroplethMap from '@/components/ChoroplethMap.vue';
import worldCountries from '@/assets/world_countries.json';
import StackedBoxesLegend from "@/components/StackedBoxesLegend";
import QueryCard from "@/components/QueryCard";
import ResultCard from "@/components/ResultCard";
import PointGroup from "@/components/PointGroup";
import IncidentCard from "@/components/IncidentCard";

const GTDProps = {
  height: {
    type: Number,
    default: 800,
  },
  width: {
    type: Number,
    default: 1200,
  },
  margin: {
    type: Object,
    default: () => ({
      top: 50,
      bottom: 50,
      left: 50,
      right: 50,
    }),
  },
  exponent: {
    type: Number,
    default: () => 0.5,
  },
};

export default {
  name: 'GTD',
  components: {IncidentCard, PointGroup, ResultCard, QueryCard, StackedBoxesLegend, ChoroplethMap},
  props: GTDProps,
  computed: {
    canvasWidth() {
      return this.width + this.margin.left + this.margin.right;
    }, // Width of the canvas + margins
    canvasHeight() {
      return this.height + this.margin.top + this.margin.bottom;
    }, // height of the canvas + margins
    mostFatal() {
      if (this.incidents.length) {
        return d3.max(this.incidents, i => i.nkill)
      } else {
        return 0;
      }
    }, // Highest casualties in single incident; all incidents
    mostCasualties() {
      return d3.max(Object.values(this.totals).map(c => c.totalCasualties));
    }, // Highest casualties in single country; all incidents
    totalCasualties() {
      return Object.values(this.totals).reduce((total, c) => total + c.totalCasualties, 0)
    }, // Total casualties of all countries, all time
    minYear() {
      return 1970
    }, // Earliest year of incidents in GTD
    maxYear() {
      return 2018
    }, // Latest year of incidents in GTD
    pointGroups() {
      const {incidents, focused, mostFatal, maxResults} = this;
      const scale = d3.scalePow()
        .exponent(2)
        .domain([0, mostFatal])
        .range([5, 10])

      return this.queries.map((q, i) => {
        let ids = q.eventids;
        if (focused) {
          ids = ids.filter((id) => incidents[id].country_txt === focused)
        }
        const points = ids.map(eventid => incidents[eventid])
          .sort((i1, i2) => i1.nkill - i2.nkill)
          .slice(0, maxResults)
          .map(incident => {
            return {
              r: scale(incident.nkill),
              fill: d3.schemeTableau10[i],
              stroke: 'black',
              'stroke-width': '.5px',
              'data-eventid': incident.eventid,
              'data-latitude': incident.latitude,
              'data-longitude': incident.longitude,
              'data-country': incident.country_txt,
              'data-city': incident.city,
              'data-nkill': incident.nkill,
              'data-r': scale(incident.nkill),
            }
          })
        return {
          name: q.name,
          points,
        }
      });
    }, // Points on the map generated by incidents
  },
  created() {
    this.center = {
      x: this.width / 2,
      y: this.height / 2,
    }
    gtd.getCountries().then(totals => {
      this.totals = totals;
      this.$refs.gtdMap.draw();
      const zoom = d3.zoom()
        .scaleExtent([1, 8])
        .on('zoom', this.onZoom);
      d3.select('svg').call(zoom)
    });
  },
  data() {
    return {
      totals: {}, /** Gets all country totals metrics and eventids **/
      features: worldCountries.features, /** GeoJSON Features **/
      focused: '', /** The focused country. Empty string if none **/
      focusIncidentId: '', // Point that user has focused on
      k: 1, /** Zoom level **/
      showQuery: true, /** Unused **/
      incidents: {},
      queries: [],
      maxResults: 50,
    };
  },
  methods: {
    handleCreated({geometries}) {
      this.viewBox = [[0, 0], [this.canvasWidth, this.canvasHeight]]
      this.colorize({geometries});
    }, // Colorize the choropleth map after rendering.
    handleClicked(e) {
      const country = d3.select(`#${e.target.id}`)
        .attr('data-name')
      if (!this.focused && country) {
        this.focused = country;
        this.zoom(e);
      } else if (this.focused !== country) {
        this.reset();
      }
      e.stopPropagation();
    }, // onclick for country
    colorize({geometries}) {
      const colorScale = d3.scalePow()
        .exponent(this.exponent)
        .domain([0, this.mostCasualties])
        .range([0, 1]);

      const transition = d3.transition()
        .duration(1050)
        .ease(d3.easeLinear);

      const countries = this.totals;
      geometries
        .transition(transition)
        .attr('fill', function (e) {
          const count = countries[e.id] ? countries[e.id].totalCasualties : 0;
          const val = colorScale(count);
          return d3.interpolateReds(val);
        });
    }, // Colorize map from totals
    zoom(e) {
      const features = worldCountries.features
        .filter((f) => f.id === e.target.id
          || f.properties.name === e.target.name);
      if (features.length < 1) {
        return;
      }
      const feature = features[0];
      const transition = d3.transition()
        .duration(1000)
        .ease(d3.easeLinear);
      const zoom = d3.zoom()
        .scaleExtent([1, 8])
        .on('zoom', this.onZoom);
      const [[x0, y0], [x1, y1]] = this.$refs.gtdMap.path.bounds(feature);
      const canvas = d3.select('#gtd-canvas');
      const zoomFactor = Math.max((x1 - x0) / this.width, (y1 - y0) / this.height);
      const scaleFactor = Math.min(8, 0.9 / zoomFactor);
      const xPrime = (x0 + x1) / -2;
      const yPrime = (y0 + y1) / -2;

      canvas.transition(transition)
        .call(zoom.transform,
          d3.zoomIdentity.translate(this.width / 2, this.height / 2)
            .scale(scaleFactor)
            .translate(xPrime, yPrime))
        .call((t, id) => {
          d3.selectAll(`.path.geometry:not([id=${id}])`)
            .transition(t)
            .style('opacity', 0.1);
        }, feature.id)


    }, // Determines where and how much to zoom then emits zoom event
    unzoom() {
      const zoom = d3.zoom()
        .scaleExtent([1, 8])
        .on('zoom', this.onZoom);
      const transition = d3.transition()
        .duration(1000)
        .ease(d3.easeLinear);
      const canvas = d3.select('#gtd-canvas');

      canvas.transition(transition)
        .call(zoom.transform, d3.zoomIdentity)
        .call((t) => {
          d3.selectAll('.path.geometry')
            .transition(t)
            .style('opacity', '100%');
        });
    }, // Emits zoom event for "unzooming"
    onZoom(e) {
      this.k = e.transform.k
      const geometries = d3.selectAll('.path.geometry');
      geometries.attr('transform', e.transform);
      geometries.attr('stroke-width', 1 / this.k);
      d3.selectAll('.point-group')
        .attr('transform', e.transform)
      const points = d3.selectAll('.point');

      points.data(points)
        .attr('r', d => {
          return d.getAttribute('data-r') / this.k
        })
        .attr('stroke-width', 1 / this.k)
    }, // onzoom handling
    reset() {
      this.features = worldCountries.features;
      this.unzoom();
      this.focused = '';
      this.focusIncidentId = '';
    }, // Reset features, zoom, and focus
    querySubmit(e, query) {
      const that = this;
      const params = {
        country: query.country === 'All' ? undefined : query.country,
        startYear: query.startYear,
        endYear: query.endYear,
        minCasualties: query.showNonFatal ? 0 : 1,
      }
      const getCountries = gtd.getCountries(params)
      const getIncidents = gtd.getIncidents(params)
      const {zoom, unzoom} = this;

      Promise.all([getCountries, getIncidents]).then(([countries, incidents]) => {
        if (params.country) {
          that.focused = params.country;
          zoom({target: {name: params.country}})
        } else {
          unzoom();
        }
        incidents.forEach(i => {
          that.incidents[i.eventid] = i
        })
        const i = that.queries.length
        const stats = Object.values(countries).reduce((total, d) => {
          total.totalIncidents += d.totalIncidents;
          total.totalCasualties += d.totalCasualties;
          total.eventids.push(...d.incidents)
          return total;
        }, {totalIncidents: 0, totalCasualties: 0, eventids: []});
        stats.casualtyAvg = stats.totalCasualties / stats.totalIncidents;
        that.queries.push({
          title: query.title,
          color: d3.schemeTableau10[i],
          ...params,
          ...stats,
        });
      })
    }, // Handle users GTD query
    formatNumber(f) {
      return f.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",")
    }, // Formats a number for commas
    mouseoverPoint(target) {
      this.focusIncidentId = d3.select(target).attr('data-eventid');
    }, // Show the incident details when a user mouses over
    dismiss() {
      this.focusIncidentId = '';
    }, // Dismiss incident details
  },
}
;
</script>
<style>
html, body {
  width: 100%;
}

#gtd-content {
  display: flex;
  flex-direction: column;
  justify-content: center;
  margin: 0 15px;
}

.gtd-row {
  display: flex;
  flex-direction: row;
  justify-content: flex-start;
  margin: 15px 0;
}

.gtd-item {
  margin: 0 10px
}

.incident-placeholder {
  height: 200px;
  width: 1200px;
  background: #000;
  opacity: .87;
  color: #ddd;
  border-radius: 15px;
  text-align: center;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  font-size: 1.5em;
}

#left-panel {
  width: 300px;
  height: 900px;
  display: flex;
  flex-direction: column;
}

#map-panel {
  position: relative;
  overflow: hidden;
  width: 900px;
  height: 700px;
}

#gtd-canvas {
  position: absolute;
  top: 0;
  left: 0;
}

#gtd-incident {
  position: absolute;
  top: 10px;
  left: 5px;
}

.slide-up-enter-active, .slide-up-leave-active {
  transition: 1s;
}

.slide-up-enter, .slide-up-leave-to {
  transform: translate(0, -250px);
}

.choropleth.canvas {
  background-color: #a1b6ec;
  overflow: hidden;
  border-radius: 15px;
}

svg text {
  fill: black;
}

.spacer {
  width: 100%;
  height: 25px;
}

.margin-bottom-small {
  margin-bottom: 5px;
}
</style>
