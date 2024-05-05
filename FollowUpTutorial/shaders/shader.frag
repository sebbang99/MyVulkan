#version 450

layout(location = 0) in vec3 fragColor;

layout(location = 0) out vec4 outColor;	// location ���� frame buffer index.

void main() {	// �� fragment�� ������ �����ϴµ�, vertex shader���� �� vertex�� ���� �����س��Ҵ�.
	outColor = vec4(fragColor, 1.0);
}