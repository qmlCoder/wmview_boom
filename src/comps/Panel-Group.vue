<style scoped>
div.title {
  /* font-size: large; */
  text-align: center;
  width: 100%;
}

div.shadow {
  box-shadow: 0 0 2px black;
}

div.root {
  background-color: white;
  display: flex;
  flex-direction: column;
  border-radius: 8px;
  padding: 5px;
  margin: 5px;
  margin-bottom: 10px;
  width: calc(100% - 10px);
  box-sizing: border-box;
  /*overflow: visible;*/
}
</style>

<template>
  <div :class="['root', { shadow: props.shadow }]">
    <div class="title">{{ props.title }}</div>
    <slot></slot>
  </div>
</template>

<script setup lang="ts">
const props = defineProps({
  title: String,
  minWidth: {
    type: String,
    default: undefined,
  },
  shadow: {
    type: Boolean,
    default: true,
  },
})
</script>
