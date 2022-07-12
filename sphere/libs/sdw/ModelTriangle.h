#pragma once

#include <glm/glm.hpp>
#include <string>
#include <array>
#include "Colour.h"
#include "TexturePoint.h"
// #include "TextureMap.h"

struct ModelTriangle {
	std::string name;
	std::array<glm::vec3, 3> vertices{};
	std::array<glm::vec3, 3> vertexNormal{};
	std::array<TexturePoint, 3> texturePoints{};
	// TextureMap texMap{};
	Colour colour{};
	glm::vec3 normal{};

	ModelTriangle();
	ModelTriangle(const glm::vec3 &v0, const glm::vec3 &v1, const glm::vec3 &v2, Colour trigColour);
	friend std::ostream &operator<<(std::ostream &os, const ModelTriangle &triangle);
};
