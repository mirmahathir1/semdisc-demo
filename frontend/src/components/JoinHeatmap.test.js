import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import JoinHeatmap from './JoinHeatmap.vue'

describe('JoinHeatmap', () => {
  it('emits path ids for a selected heatmap group', async () => {
    const wrapper = mount(JoinHeatmap, {
      props: {
        paths: [
          { id: 'p1', nodeCount: 2, tupleCount: 3 },
          { id: 'p2', nodeCount: 2, tupleCount: 4 }
        ]
      }
    })

    await wrapper.find('button').trigger('click')

    expect(wrapper.emitted('select')[0][0]).toContain('p1')
  })

  it('renders tuple bins as rows and node counts as columns', () => {
    const wrapper = mount(JoinHeatmap, {
      props: {
        paths: [
          { id: 'p1', nodeCount: 2, tupleCount: 3 },
          { id: 'p2', nodeCount: 4, tupleCount: 12 }
        ]
      }
    })

    expect(wrapper.text()).toContain('Tuple count')
    expect(wrapper.findAll('.heatmap-header').some((header) => header.text() === '2')).toBe(true)
    expect(wrapper.findAll('.heatmap-header').some((header) => header.text() === '4')).toBe(true)
    expect(wrapper.findAll('.heatmap-bin').length).toBeGreaterThan(0)
  })
})
