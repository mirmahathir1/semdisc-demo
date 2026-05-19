import { createRouter, createWebHistory } from 'vue-router'
import NavigateView from '../views/NavigateView.vue'
import QueryInputView from '../views/QueryInputView.vue'
import RetrievedPathsView from '../views/RetrievedPathsView.vue'

export default createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'navigate', component: NavigateView },
    { path: '/query', name: 'query', component: QueryInputView },
    { path: '/retrieved', name: 'retrieved', component: RetrievedPathsView }
  ]
})

