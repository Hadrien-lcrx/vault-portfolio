<template>
  <div :class="`cardcontrol mdc-card ${classs}`" >
    <div class="cardcontrol content">
      <slot/>
    </div>
    <div class="cardcontrol footer"> </div>
  </div>
</template>
<script>
export default {
  name: 'CardControl',
  props: {
    classs: {
      type: String,
      default: () => '',
    },
  },
  mounted() {
  }
}
</script>
<style lang="scss">
.cardcontrol {
  width: 100%;
}

.cardcontrol .nav {
  display: inline-flex;
  flex-direction: row;
  justify-content: space-between;
  font-size: .8em;
}

.iconbar {
  display: inline-flex;
  flex-direction: row;
  justify-content: flex-end;
}

.clickable {
  cursor: pointer;
}

</style>
