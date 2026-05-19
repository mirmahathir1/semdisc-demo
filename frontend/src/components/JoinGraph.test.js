import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import JoinGraph from './JoinGraph.vue'

describe('JoinGraph', () => {
  it('colors equi joins blue and semantic joins orange', () => {
    const wrapper = mount(JoinGraph, {
      props: {
        defaultOpen: true,
        graph: {
          nodes: [
            { id: 'customers', label: 'customers' },
            { id: 'orders', label: 'orders' }
          ],
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

  it('can hide the edge legend when another view owns the legend', () => {
    const wrapper = mount(JoinGraph, {
      props: {
        defaultOpen: true,
        showLegend: false,
        path: {
          nodes: ['customers', 'orders'],
          edges: [
            {
              left: 'customers',
              right: 'orders',
              key: 'customer_id',
              rightKey: 'customer_id',
              joinType: 'semantic'
            }
          ]
        }
      }
    })

    expect(wrapper.text()).not.toContain('Semantic join')
  })

  it('shows only the graph-surviving edge for each path node pair', () => {
    const wrapper = mount(JoinGraph, {
      props: {
        defaultOpen: true,
        graphEdges: [
          {
            left: 'customers',
            right: 'orders',
            key: 'customer_no',
            rightKey: 'customer_no',
            joinType: 'equi'
          }
        ],
        path: {
          nodes: ['customers', 'orders'],
          edges: [
            {
              left: 'customers',
              right: 'orders',
              key: 'customer_id',
              rightKey: 'customer_id',
              joinType: 'semantic'
            },
            {
              left: 'customers',
              right: 'orders',
              key: 'customer_no',
              rightKey: 'customer_no',
              joinType: 'equi'
            }
          ]
        }
      }
    })

    expect(wrapper.findAll('path')).toHaveLength(1)
    expect(wrapper.text()).toContain('customer_no / customer_no')
    expect(wrapper.text()).not.toContain('customer_id / customer_id')
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

  it('highlights tables when selected query columns use display labels', () => {
    const wrapper = mount(JoinGraph, {
      props: {
        defaultOpen: true,
        highlightColumn: 'Order ID',
        tableColumns: {
          query: ['Order ID'],
          orders: ['order_id', 'customer_id'],
          shipments: ['order_id'],
          customers: ['customer_id']
        },
        path: {
          nodes: ['query', 'orders', 'shipments', 'customers'],
          edges: [
            {
              left: 'orders',
              right: 'shipments',
              key: 'order_id',
              rightKey: 'order_id',
              joinType: 'semantic'
            },
            {
              left: 'orders',
              right: 'customers',
              key: 'customer_id',
              rightKey: 'customer_id',
              joinType: 'semantic'
            }
          ]
        }
      }
    })

    const rects = wrapper.findAll('rect')

    expect(rects[0].attributes('fill')).toBe('#dbeafe')
    expect(rects[1].attributes('fill')).toBe('#dcfce7')
    expect(rects[2].attributes('fill')).toBe('#dcfce7')
    expect(rects[3].attributes('fill')).toBe('#f8fafc')
  })

  it('uses light green for any query-column match and stronger green for the selected column', () => {
    const wrapper = mount(JoinGraph, {
      props: {
        defaultOpen: true,
        highlightColumn: 'Order ID',
        queryColumns: ['Order ID', 'Customer ID'],
        tableColumns: {
          orders: ['order_id', 'customer_id'],
          shipments: ['order_id'],
          customers: ['customer_id'],
          products: ['product_id']
        },
        path: {
          nodes: ['orders', 'shipments', 'customers', 'products'],
          edges: [
            {
              left: 'orders',
              right: 'shipments',
              key: 'order_id',
              rightKey: 'order_id',
              joinType: 'semantic'
            },
            {
              left: 'orders',
              right: 'customers',
              key: 'customer_id',
              rightKey: 'customer_id',
              joinType: 'semantic'
            }
          ]
        }
      }
    })

    const rects = wrapper.findAll('rect')

    expect(rects[0].attributes('fill')).toBe('#dcfce7')
    expect(rects[1].attributes('fill')).toBe('#dcfce7')
    expect(rects[2].attributes('fill')).toBe('#f0fdf4')
    expect(rects[2].attributes('stroke')).toBe('#86efac')
    expect(rects[3].attributes('fill')).toBe('#f8fafc')
  })
})
