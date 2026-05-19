import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import JoinGraph from './JoinGraph.vue'

describe('JoinGraph', () => {
  it('colors equi joins blue and semantic joins orange', () => {
    const wrapper = mount(JoinGraph, {
      props: {
        defaultOpen: true,
        path: {
          nodes: ['customers', 'orders'],
          edges: [
            {
              left: 'customers',
              right: 'orders',
              key: 'country',
              rightKey: 'ship_country',
              joinType: 'semantic'
            },
            {
              left: 'customers',
              right: 'orders',
              key: 'customer_id',
              rightKey: 'customer_id',
              joinType: 'equi'
            }
          ]
        }
      }
    })

    const strokes = wrapper.findAll('path').map((p) => p.attributes('stroke'))

    expect(strokes).toContain('#f97316')
    expect(strokes).toContain('#2563eb')
    expect(wrapper.text()).toContain('Semantic join')
    expect(wrapper.text()).toContain('Equi-join')
    expect(wrapper.text()).toContain('1 equi')
    expect(wrapper.text()).toContain('1 semantic')
  })

  it('fans parallel edges between the same node pair apart instead of stacking them', () => {
    const wrapper = mount(JoinGraph, {
      props: {
        defaultOpen: true,
        graph: {
          nodes: [
            { id: 'orders', label: 'orders' },
            { id: 'shipments', label: 'shipments' }
          ],
          edges: [
            {
              left: 'orders',
              right: 'shipments',
              key: 'order_id',
              rightKey: 'order_id',
              joinType: 'equi'
            },
            {
              left: 'orders',
              right: 'shipments',
              key: 'preferred_carrier',
              rightKey: 'carrier',
              joinType: 'semantic'
            }
          ]
        }
      }
    })

    const paths = wrapper.findAll('path')
    expect(paths).toHaveLength(2)

    // both edges are curves on the same node pair, fanned to distinct bows
    expect(paths[0].attributes('d')).toMatch(/^M .* Q .* /)
    expect(paths[1].attributes('d')).toMatch(/^M .* Q .* /)
    expect(paths[0].attributes('d')).not.toBe(paths[1].attributes('d'))

    expect(wrapper.text()).toContain('order_id / order_id')
    expect(wrapper.text()).toContain('preferred_carrier / carrier')
  })

  it('highlights path tables containing the selected query column in green', () => {
    const wrapper = mount(JoinGraph, {
      props: {
        defaultOpen: true,
        highlightColumn: 'country',
        tableColumns: {
          customers: ['customer_id', 'country'],
          orders: ['order_id', 'customer_id']
        },
        path: {
          nodes: ['customers', 'orders'],
          edges: [
            {
              left: 'customers',
              right: 'orders',
              key: 'customer_id',
              rightKey: 'customer_id',
              joinType: 'equi'
            }
          ]
        }
      }
    })

    const rects = wrapper.findAll('rect')

    expect(rects[0].attributes('fill')).toBe('#dcfce7')
    expect(rects[0].attributes('stroke')).toBe('#16a34a')
    expect(rects[1].attributes('fill')).toBe('#f8fafc')
  })
})
