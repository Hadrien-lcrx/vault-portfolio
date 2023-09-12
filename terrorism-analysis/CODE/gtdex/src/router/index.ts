import Vue from 'vue';
import VueRouter, { RouteConfig } from 'vue-router';
import GTD from '../views/GTD.vue';
import Bloom from '../views/Bloom.vue';

Vue.use(VueRouter);

const routes: Array<RouteConfig> = [
  {
    path: '/',
    name: 'GTD',
    component: GTD,
    props: {
      height: 600,
      width: 800,
    },
  },
  {
    path: '/bloom',
    name: 'Bloom',
    component: Bloom,
    props: {}
  },
];

const router = new VueRouter({
  mode: 'history',
  base: process.env.BASE_URL,
  routes,
});

export default router;
