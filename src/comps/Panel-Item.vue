<style scoped>
div.main {
  display: flex;
  align-items: stretch;
  margin: 3px 0px;
  min-height: 2rem;
  width: 100%;
  background-color: white;
  flex-wrap: nowrap;
}

div.main:hover {
  background-color: aliceblue;
}

div.title {
  width: 6rem;
  font-size: small;
  cursor: pointer;
  background-color: #f8f9fa;
  display: flex;
  align-items: center;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
  position: relative;
}

div.title_over:hover span {
  display: inline-block;
  animation: scroll-text 3s linear infinite;
}

@keyframes scroll-text {
  0% {
    transform: translateX(0);
  }

  100% {
    transform: translateX(-100%);
  }
}

div.slot {
  width: calc(100% - 6rem);
}

div.row {
  display: flex;
  flex-direction: row;
  flex-wrap: nowrap;
  /* justify-content: space-around; */
  align-items: center;
  gap: 3px;
}
</style>

<template>
  <div class="main">
    <div class="title" ref="titleDom" :class="{ title_over: overflow }">
      <span v-if="props.tip == ''">{{ props.title }}</span>
      <span v-else>
        <el-tooltip :content="props.tip" placement="top" :enterable="false">
          {{ props.title }}
        </el-tooltip>
      </span>
    </div>
    <div :class="{ slot: true, row: props.row }">
      <slot></slot>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useTemplateRef, computed } from 'vue'
import { ElTooltip } from 'element-plus'
const props = defineProps({
  title: String,
  tip: {
    type: String,
    optional: true,
    default: '',
  },
  row: {
    type: Boolean,
    optional: true,
    default: true,
  },
})

const titleDom = useTemplateRef<HTMLDivElement>('titleDom')
const overflow = computed(() => {
  if (!titleDom.value) return false
  return titleDom.value.scrollWidth > titleDom.value.clientWidth
})
</script>
