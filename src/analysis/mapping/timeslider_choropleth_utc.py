from folium.plugins import TimeSliderChoropleth
from branca.element import Template

TimeSliderChoropleth._template = Template("""
        {% macro script(this, kwargs) %}
        {
            let timestamps = {{ this.timestamps|tojson }};
            let styledict = {{ this.styledict|tojson }};
            let current_timestamp = timestamps[{{ this.init_timestamp }}];

            function formatDate(date) {
               var newdate = new moment(date).utc();
               return newdate.format({{this.date_format|tojson}});
            }

            let slider_body = d3.select("body").insert("div", "div.folium-map")
                .attr("id", "slider_{{ this.get_name() }}");
            $("#slider_{{ this.get_name() }}").hide();
            slider_body.append("output")
                .style('font-size', '18px')
                .style('text-align', 'center')
                .style('font-weight', '500%')
                .style('margin', '5px');
            slider_body.append("input")
                .attr("type", "range")
                .attr("width", "100px")
                .attr("min", 0)
                .attr("max", timestamps.length - 1)
                .attr("value", {{ this.init_timestamp }})
                .attr("step", "1")
                .style('align', 'center');

            let datestring = formatDate(parseInt(current_timestamp)*1000);
            d3.select("#slider_{{ this.get_name() }} > output").text(datestring);

            let fill_map = function(){
                // 1. Clear opacity across all grid features
                d3.selectAll('path[id^="{{ this.get_name() }}-feature-"]')
                    .style('fill-opacity', 0);

                // 2. Color active features for current timestep
                for (var feature_id in styledict){
                    let style = styledict[feature_id];
                    
                    if (style && current_timestamp in style){
                        var fillColor = style[current_timestamp] || '#000000';
                        d3.selectAll('#{{ this.get_name() }}-feature-' + feature_id)
                            .attr('fill', fillColor)
                            .style('fill-opacity', 0.65);
                    }
                }
            }

            // Hook for dynamic plume temperature updates
            window.updateTimeSliderStyle_{{ this.get_name() }} = function(newStyleDict) {
                styledict = newStyleDict;
                fill_map();
            };

            d3.select("#slider_{{ this.get_name() }} > input").on("input", function() {
                current_timestamp = timestamps[this.value];
                let datestring = formatDate(parseInt(current_timestamp)*1000);
                d3.select("#slider_{{ this.get_name() }} > output").text(datestring);
                fill_map();
            });

            let onEachFeature;
            {% if this.highlight %}
                 onEachFeature = function(feature, layer) {
                    layer.on({
                        mouseout: function(e) {
                            if (styledict[e.target.feature.id] && current_timestamp in styledict[e.target.feature.id]){
                                d3.selectAll('#{{ this.get_name() }}-feature-'+e.target.feature.id).style('fill-opacity', 0.65);
                            }
                        },
                        mouseover: function(e) {
                            if (styledict[e.target.feature.id] && current_timestamp in styledict[e.target.feature.id]){
                                d3.selectAll('#{{ this.get_name() }}-feature-'+e.target.feature.id).style('fill-opacity', 1);
                            }
                        },
                        click: function(e) {
                            {{this._parent.get_name()}}.fitBounds(e.target.getBounds());
                        }
                    });
                };
            {% endif %}

            var {{ this.get_name() }} = L.geoJson(
                {{ this.data|tojson }},
                {
                    style: function(feature) {
                        return {
                            fillColor: '#000000',
                            fillOpacity: 0,
                            stroke: false,
                            weight: 0
                        };
                    },
                    onEachFeature: onEachFeature
                }
            );

            let onOverlayAdd = function(e) {
                {{ this.get_name() }}.eachLayer(function (layer) {
                    if (layer._path && layer.feature && layer.feature.id !== undefined) {
                        layer._path.id = '{{ this.get_name() }}-feature-' + layer.feature.id;
                    }
                });

                $("#slider_{{ this.get_name() }}").show();

                d3.selectAll('path')
                    .attr('stroke', 'none')
                    .attr('stroke-width', 0)
                    .attr('fill-opacity', 0);

                fill_map();
            }

            {{ this.get_name() }}.on('add', onOverlayAdd);
            {{ this.get_name() }}.on('remove', function() {
                $("#slider_{{ this.get_name() }}").hide();
            });

            {%- if this.show %}
            {{ this.get_name() }}.addTo({{ this._parent.get_name() }});
            $("#slider_{{ this.get_name() }}").show();
            {%- endif %}
        }
        {% endmacro %}
""")