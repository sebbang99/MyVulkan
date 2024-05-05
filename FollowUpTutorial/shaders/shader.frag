#version 450

layout(location = 0) in vec3 fragColor;

layout(location = 0) out vec4 outColor;	// location 값은 frame buffer index.

void main() {	// 각 fragment의 색깔을 지정하는데, vertex shader에서 준 vertex의 색을 보간해놓았다.
	outColor = vec4(fragColor, 1.0);
}