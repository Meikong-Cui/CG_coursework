#include <CanvasTriangle.h>
#include <CanvasPoint.h>
#include <Colour.h>
#include <DrawingWindow.h>
#include <ModelTriangle.h>
#include <RayTriangleIntersection.h>
#include <TextureMap.h>
#include <TexturePoint.h>
#include <Utils.h>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <glm/glm.hpp>

#define WIDTH 320
#define HEIGHT 240
#define PI 3.14159265

struct Settings {
	bool proximity;
	bool incident;
	bool specular;
	bool texture;
	bool phongShader;
	bool softShadow;
	float specularStrength;
	float ambient;
	Settings() = default;
	Settings(bool proximity, bool incident, bool specular, bool texture, bool phongShader, bool softShadow, float specularStrength, float ambient):
		proximity(proximity), incident(incident), specular(specular), texture(texture), phongShader(phongShader), softShadow(softShadow), specularStrength(specularStrength), ambient(ambient) {};
};

glm::mat4 translate_matrix(float dx, float dy, float dz) {
	glm::mat4 translate = glm::mat4(1.0f, 0.0f, 0.0f, 0.0f,
									0.0f, 1.0f, 0.0f, 0.0f,
									0.0f, 0.0f, 1.0f, 0.0f,
									dx, dy, dz, 1.0f);
								
	return translate;
}

glm::mat4 rotate_matrix(float x_angle, float y_angle, float z_angle) {
	float ax = x_angle * PI / 180;
	float ay = y_angle * PI / 180;
	float az = z_angle * PI / 180;
	glm::mat4 result;
	glm::mat4 rx_mat = glm::mat4(1.0f, 0.0f, 0.0f, 0.0f,
								 0.0f, cos(ax), sin(ax), 0.0f,
								 0.0f, -sin(ax), cos(ax), 0.0f,
								 0.0f, 0.0f, 0.0f, 1.0f);

	glm::mat4 ry_mat = glm::mat4(cos(ay), 0.0f, -sin(ay), 0.0f,
								 0.0f, 1.0f, 0.0f, 0.0f,
								 sin(ay), 0.0f, cos(ay), 0.0f,
								 0.0f, 0.0f, 0.0f, 1.0f);

	glm::mat4 rz_mat = glm::mat4(cos(az), sin(az), 0.0f, 0.0f,
								 -sin(az), cos(az), 0.0f, 0.0f,
								 0.0f, 0.0f, 1.0f, 0.0f,
								 0.0f, 0.0f, 0.0f, 1.0f);
			
	result = rx_mat * ry_mat * rz_mat;
	return result;
}

glm::mat4 lookAt(glm::mat4 &camera, glm::vec3 object) {
	glm::vec3 z = glm::normalize(glm::vec3(camera[3] / camera[3].w) - object);
	glm::vec3 x = glm::normalize(glm::cross(glm::vec3(0.0, 1.0, 0.0), z));
	glm::vec3 y = glm::normalize(glm::cross(z, x));
	return glm::mat4(
		glm::vec4(x, 0.0),
		glm::vec4(y, 0.0),
		glm::vec4(z, 0.0),
		glm::vec4(camera[3])
	);
}

std::vector<float> interpolateSingleFloats(float from, float to, int numberOfValues) {
	std::vector<float> result;
	if (numberOfValues < 2) {
		result.push_back(from);
		return result;
	} else {	
		result.push_back(from);
		float temp = from;
		float step = (to - from)/(numberOfValues - 1);
		for (int x = 0; x < numberOfValues - 1; x++) {
			temp += step;
			result.push_back(temp);
		}
	}
	return result;
}

std::vector<CanvasPoint> interpolateCanavasPoint(CanvasPoint from, CanvasPoint to, int numberOfValues) {
	std::vector<CanvasPoint> result;
	if (numberOfValues < 2) {
		result.push_back(from);
	} else {
		// result.push_back(from);
        float stepX = (to.x - from.x)/(numberOfValues - 1);
		float stepY = (to.y - from.y)/(numberOfValues - 1);
		float stepZ = (to.depth - from.depth)/(numberOfValues - 1);
        for (int i = 0; i < numberOfValues; i++) {
			CanvasPoint p(round(from.x + i * stepX), round(from.y + i * stepY), (from.depth + i * stepZ));
			result.push_back(p);
		}
	}
	return result;
}

std::vector<TexturePoint> interpolateTexturePoint(TexturePoint from, TexturePoint to, int numberOfValues) {
	std::vector<TexturePoint> result;
	if (numberOfValues < 2) {
		result.push_back(from);
	} else {
        // float stepX = (to.x - from.x)/(numberOfValues - 1);
		// float stepY = (to.y - from.y)/(numberOfValues - 1);
		float txDiff = (to.x - from.x)/(numberOfValues - 1);
		float tyDiff = (to.y - from.y)/(numberOfValues - 1);
		// float stepZ = (to.depth - from.depth)/(numberOfValues - 1);
        for (float i = 0.0f; i < numberOfValues - 1; i++) {
			// CanvasPoint p(round(from.x + i * stepX), round(from.y + i * stepY), (from.depth + i * stepZ));
			TexturePoint t((from.x + i * txDiff), (from.y + i * tyDiff));
			// p.texturePoint = t;
			result.push_back(t);
		}
	}
	return result;
}

CanvasPoint getCanvasIntersectionPoint(glm::mat4 cameraPosition, glm::vec3 vertexPosition, float focalLength, int scale) {
	glm::mat4 ori = glm::mat4(cameraPosition[0], cameraPosition[1], cameraPosition[2], glm::vec4(0.0, 0.0, 0.0, 1.0f));
	glm::vec4 camera = cameraPosition[3];
	glm::vec3 vertex = glm::vec3((glm::vec4(vertexPosition, 1.0f) - camera) * ori);
	double u = -focalLength * vertex.x * scale / vertex.z + WIDTH/2;
	double v = focalLength * vertex.y * scale / vertex.z + HEIGHT/2;
	CanvasPoint result(round(u), round(v), 1/vertex.z);
	return result;
}

RayTriangleIntersection getClosestIntersection(glm::vec3 eyePos, glm::vec3 rayDirection, std::vector<ModelTriangle> &triangles) {
	RayTriangleIntersection intersect;
	intersect.distanceFromCamera = std::numeric_limits<float>::infinity();
	float small = 0.0001;
	for(size_t i=0; i<triangles.size(); i++) {
		glm::vec3 e0 = triangles[i].vertices[1] - triangles[i].vertices[0];
		glm::vec3 e1 = triangles[i].vertices[2] - triangles[i].vertices[0];
		glm::vec3 SPVector = eyePos - triangles[i].vertices[0];
		glm::mat3 DEMatrix(-rayDirection, e0, e1);
		glm::vec3 possibleSolution = glm::inverse(DEMatrix) * SPVector;
		if(possibleSolution.x > small && possibleSolution.x < intersect.distanceFromCamera && 
		   possibleSolution.y >= 0.0 && possibleSolution.y <= 1.0 && 
		   possibleSolution.z >= 0.0 && possibleSolution.z <= 1.0 && 
		   (possibleSolution.y + possibleSolution.z) <= 1.0) {
			intersect.intersectionPoint = possibleSolution;
			intersect.distanceFromCamera = possibleSolution.x;
			intersect.intersectedTriangle = triangles[i];
			intersect.triangleIndex = i;
		}
	}
	return intersect;
}

void drawLine(CanvasPoint from, CanvasPoint to, float** zBuffer, Colour colour, DrawingWindow &window) {
	float xDiff = ceil(to.x) - floor(from.x);
	float yDiff = to.y - from.y;
	float numOfSteps = fmax(fmax(fabs(xDiff), fabs(yDiff)), 1);
	float xStepSize = xDiff / numOfSteps;
	float yStepSize = yDiff / numOfSteps;
	float zStepSize = (to.depth - from.depth) / numOfSteps;
	uint32_t c = (255 << 24) + (int(colour.red) << 16) + (int(colour.green) << 8) + int(colour.blue);	
	for (float i = 0.0; i < (numOfSteps + 1); i++) {
		float z = from.depth + zStepSize * i;
		float x = round(from.x + xStepSize * i);
		float y = round(from.y + yStepSize * i);
		if(y > 0 && y < HEIGHT && x > 0 && x < WIDTH && (z) <= zBuffer[int(y)][int(x)]) {
			zBuffer[int(y)][int(x)] = z;
			window.setPixelColour(x, y, c);
		}
		// else continue;
	}
}

void drawTexLine(CanvasPoint from, CanvasPoint to, float** zBuffer, TextureMap &texture, DrawingWindow &window) {
	float xDiff = ceil(to.x) - floor(from.x);
	float yDiff = to.y - from.y;
	float numOfSteps = fmax(fmax(fabs(xDiff), fabs(yDiff)), 1);
	float xStepSize = xDiff / numOfSteps;
	float yStepSize = yDiff / numOfSteps;
	float txStep = (to.texturePoint.x - from.texturePoint.x)/numOfSteps;
	float tyStep = (to.texturePoint.y - from.texturePoint.y)/numOfSteps;
	float zStepSize = (to.depth - from.depth) / numOfSteps;
	for (float i = 0.0; i < (numOfSteps + 1); i++) {
		float z = from.depth + zStepSize * i;
		float x = round(from.x + xStepSize * i);
		float y = round(from.y + yStepSize * i);
		if(y > 0 && y < HEIGHT && x > 0 && x < WIDTH && (z) <= zBuffer[int(y)][int(x)]) {
			zBuffer[int(y)][int(x)] = z;
			int p = int(round(from.texturePoint.x + i * txStep) + (texture.width * round(from.texturePoint.y + i * tyStep)));
			window.setPixelColour(int(x), int(y), texture.pixels[p]);
		}
		// else continue;
	}
}

void drawStrokedTriangle(CanvasTriangle triangle, float** zBuffer, Colour colour, DrawingWindow &window) {
	drawLine(triangle[0], triangle[1], zBuffer, colour, window);
	drawLine(triangle[1], triangle[2], zBuffer, colour, window);
	drawLine(triangle[2], triangle[0], zBuffer, colour, window);
}

bool ascendSort(CanvasPoint a, CanvasPoint b) {return a.y < b.y;}

void drawFilledTriangle(CanvasTriangle triangle, Colour colour, float** zBuffer, DrawingWindow &window) {
	// sort points of triangle by y value.
	for(int i = 0; i < 2; i++) {
		for(int j = 0; j < 2; j++) {
			if(triangle[j].y > triangle[j+1].y) std::swap(triangle[j], triangle[j+1]);
		}
	}

	float fourthX = triangle[0].x + ((triangle[1].y - triangle[0].y) / (triangle[2].y - triangle[0].y)) * (triangle[2].x-triangle[0].x);
	float fourthZ = triangle[0].depth + ((triangle[1].y - triangle[0].y) / (triangle[2].y - triangle[0].y)) * (triangle[2].depth-triangle[0].depth);
	CanvasPoint fourthP(fourthX, triangle[1].y, fourthZ);

	int numOfSteps = triangle[1].y - triangle[0].y + 1;	
	std::vector<CanvasPoint> l0m = interpolateCanavasPoint(triangle[0], fourthP, numOfSteps);
	std::vector<CanvasPoint> l01 = interpolateCanavasPoint(triangle[0], triangle[1], numOfSteps);
	if(fourthX < triangle[1].x) {
		for(int index = 0; index < numOfSteps; index++) {
			if(l0m[index].y < 0 || l0m[index].y >= HEIGHT) continue;
			l0m[index].x = std::fmax(l0m[index].x, 0);
			l01[index].x = std::fmin(l01[index].x, WIDTH);
			drawLine(l0m[index], l01[index], zBuffer, colour, window);
		}
	} else {
		for(int index = 0; index < numOfSteps; index++) {
			if(l0m[index].y < 0 || l0m[index].y >= HEIGHT) continue;
			l01[index].x = std::fmax(l01[index].x, 0);
			l0m[index].x = std::fmin(l0m[index].x, WIDTH);
			drawLine(l01[index], l0m[index], zBuffer, colour, window);
		}
	}
	int numOfSteps2 = triangle[2].y - triangle[1].y + 1;
	std::vector<CanvasPoint> lm2 = interpolateCanavasPoint(fourthP, triangle[2], numOfSteps2);
	std::vector<CanvasPoint> l12 = interpolateCanavasPoint(triangle[1], triangle[2], numOfSteps2);
	if(fourthX < triangle[1].x) {
		for(int index = 0; index < numOfSteps2; index++) {
			if(lm2[index].y < 0 || lm2[index].y >= HEIGHT) continue;
			lm2[index].x = std::fmax(lm2[index].x, 0);
			l12[index].x = std::fmin(l12[index].x, WIDTH);
			drawLine(lm2[index], l12[index], zBuffer, colour, window);
		}
	} else {
		for(int index = 0; index < numOfSteps2; index++) {
			if(lm2[index].y < 0 || lm2[index].y >= HEIGHT) continue;
			l12[index].x = std::fmax(l12[index].x, 0);
			lm2[index].x = std::fmin(lm2[index].x, WIDTH);
			drawLine(l12[index], lm2[index], zBuffer, colour, window);
		}
	}

	// draw white stroked triangle.
	// Colour white(255, 255, 255);
	// drawLine(triangle[0], triangle[1], zBuffer, colour, window);
	// drawLine(triangle[1], triangle[2], zBuffer, colour, window);
	// drawLine(triangle[2], triangle[0], zBuffer, colour, window);
}

void drawTexture(CanvasTriangle triangle, float** zBuffer, TextureMap &texture, DrawingWindow &window) {
	// sort triangle
	for(int i = 0; i < 2; i++) {
		for(int j = 0; j < 2; j++) {
			if(triangle[j].y > triangle[j+1].y) std::swap(triangle[j], triangle[j+1]);
		}
	}

	// float ratio = (triangle[1].y - triangle[0].y) / (triangle[2].y - triangle[0].y);
	// float fourthX = triangle[0].x + ratio * (triangle[2].x-triangle[0].x);
	// float fourthZ = triangle[0].depth + ratio * (triangle[2].depth-triangle[0].depth);
	// CanvasPoint fourthP(fourthX, triangle[1].y, fourthZ);
	// float midTexX = triangle[0].texturePoint.x + ratio * (triangle[2].texturePoint.x-triangle[0].texturePoint.x);
	// float midTexY = triangle[0].texturePoint.y + ratio * (triangle[2].texturePoint.y-triangle[0].texturePoint.y);
	// TexturePoint fourthT(midTexX, midTexY);
	// fourthP.texturePoint = fourthT;
	// int numOfSteps = triangle[1].y - triangle[0].y + 1;
	// std::vector<CanvasPoint> l0m = interpolateTexturePoint(triangle[0], fourthP, numOfSteps);
	// std::vector<CanvasPoint> l01 = interpolateTexturePoint(triangle[0], triangle[1], numOfSteps);
	// if(fourthX < triangle[1].x) {
	// 	for(int index = 0; index < numOfSteps; index++) {
	// 		if(l0m[index].y < 0 || l0m[index].y >= HEIGHT) continue;
	// 		l0m[index].x = std::fmax(l0m[index].x, 0);
	// 		l01[index].x = std::fmin(l01[index].x, WIDTH);
	// 		drawTexLine(l0m[index], l01[index], zBuffer, texture, window);
	// 	}
	// } else {
	// 	for(int index = 0; index < numOfSteps; index++) {
	// 		if(l0m[index].y < 0 || l0m[index].y >= HEIGHT) continue;
	// 		l01[index].x = std::fmax(l01[index].x, 0);
	// 		l0m[index].x = std::fmin(l0m[index].x, WIDTH);
	// 		drawTexLine(l01[index], l0m[index], zBuffer, texture, window);
	// 	}
	// }
	// int numOfSteps2 = triangle[2].y - triangle[1].y + 1;
	// std::vector<CanvasPoint> lm2 = interpolateTexturePoint(fourthP, triangle[2], numOfSteps2);
	// std::vector<CanvasPoint> l12 = interpolateTexturePoint(triangle[1], triangle[2], numOfSteps2);
	// if(fourthX < triangle[1].x) {
	// 	for(int index = 0; index < numOfSteps2; index++) {
	// 		if(lm2[index].y < 0 || lm2[index].y >= HEIGHT) continue;
	// 		lm2[index].x = std::fmax(lm2[index].x, 0);
	// 		l12[index].x = std::fmin(l12[index].x, WIDTH);
	// 		drawTexLine(lm2[index], l12[index], zBuffer, texture, window);
	// 	}
	// } else {
	// 	for(int index = 0; index < numOfSteps2; index++) {
	// 		if(lm2[index].y < 0 || lm2[index].y >= HEIGHT) continue;
	// 		l12[index].x = std::fmax(l12[index].x, 0);
	// 		lm2[index].x = std::fmin(lm2[index].x, WIDTH);
	// 		drawTexLine(l12[index], lm2[index], zBuffer, texture, window);
	// 	}
	// }

	// shade the top part triangle with texture.	
	float ratio = (triangle[1].y - triangle[0].y) / (triangle[2].y - triangle[0].y);
	float fourthX = triangle[0].x + ratio * (triangle[2].x-triangle[0].x);
	// float fourthZ = triangle[0].depth + ratio * (triangle[2].depth-triangle[0].depth);
	// CanvasPoint fourthP(fourthX, triangle[1].y, fourthZ);
	float midTexX = triangle[0].texturePoint.x + ratio * (triangle[2].texturePoint.x-triangle[0].texturePoint.x);
	float midTexY = triangle[0].texturePoint.y + ratio * (triangle[2].texturePoint.y-triangle[0].texturePoint.y);
	TexturePoint midTexturePoint(midTexX, midTexY);
	std::vector<TexturePoint> top01Texture = interpolateTexturePoint(triangle[0].texturePoint, triangle[1].texturePoint, int(triangle[1].y-triangle[0].y));
	std::vector<TexturePoint> top0mTexture = interpolateTexturePoint(triangle[0].texturePoint, midTexturePoint, int(triangle[1].y-triangle[0].y));
	float top0m = fourthX - triangle[0].x;
	float top0mStepSize = top0m / fabs(triangle[1].y - triangle[0].y);
	float top01 = triangle[1].x - triangle[0].x;
	float top01StepSize = top01 / fabs(triangle[1].y - triangle[0].y);
	float x1 = triangle[0].x;
	int bound = 0;
	for(int y = 0; y < int(triangle[1].y-triangle[0].y); y++) {
		bound = int(round(y * (fabs(top0mStepSize) + fabs(top01StepSize))));
		if(triangle[1].x < triangle[2].x) x1 = x1 + top01StepSize;
		else x1 = x1 + top0mStepSize;		// add more comparation, make sure every situation is concerned
		std::vector<TexturePoint> line = interpolateTexturePoint(top01Texture[y], top0mTexture[y], bound);
		for (int x = 0; x < bound; x++) {
			int i = round(line[x].x) + (texture.width * round(line[x].y));
			window.setPixelColour(round(x1+x), round(triangle[0].y+y), texture.pixels[i]);
		}
	}
	// shade the bottom part triangle
	std::vector<TexturePoint> bot12Texture = interpolateTexturePoint(triangle[1].texturePoint, triangle[2].texturePoint, int(triangle[2].y - triangle[1].y));
	std::vector<TexturePoint> botm2Texture = interpolateTexturePoint(midTexturePoint, triangle[2].texturePoint, int(triangle[2].y - triangle[1].y));
	float botm2 = triangle[2].x - fourthX;
	float botm2StepSize = botm2 / fabs(triangle[2].y - triangle[1].y);
	float bot12 = triangle[2].x - triangle[1].x;
	float bot12StepSize = bot12 / fabs(triangle[2].y - triangle[1].y); 
	for(int y = 0; y < int(triangle[2].y-triangle[1].y); y++) {
		if(triangle[1].x < triangle[2].x) x1 = x1 + bot12StepSize;
		else x1 = x1 + botm2StepSize;
		std::vector<TexturePoint> line = interpolateTexturePoint(bot12Texture[y], botm2Texture[y], bound);
		for (int x = 0; x < bound; x++) {
			int i = round(line[x].x) + (texture.width * round(line[x].y));
			window.setPixelColour(round(x1+x), round(triangle[1].y+y), texture.pixels[i]);
		}
		bound = bound - round(fabs(bot12StepSize - botm2StepSize));
	}

	Colour white(255, 255, 255);
	drawLine(triangle[0], triangle[1], zBuffer, white, window);
	drawLine(triangle[1], triangle[2], zBuffer, white, window);
	drawLine(triangle[2], triangle[0], zBuffer, white, window);
}

std::vector<ModelTriangle> readObj(float scale) {
	// std::ifstream mtl("/home/go19297/cg/CGLab/src/cornell-box.mtl", std::ifstream::in);
	// std::ifstream mtl("../src/textured-cornell-box.mtl", std::ifstream::in);
	std::ifstream mtl("./src/cornell-box.mtl", std::ifstream::in);
    std::string line1 = "";
	std::map<std::string, Colour> colourMap;
	while (std::getline(mtl, line1)) {
		if (line1.empty()) continue;

		if (line1[0] == 'n') {
			std::string dummy, material;
			float r, g, b;
			std::stringstream ss(line1);
			ss >> dummy >> material;
			std::string line2 = "";
			std::getline(mtl, line2);
			std::stringstream ss2(line2);
			ss2 >> dummy >> r >> g >> b;
			Colour c1(round(r*255), round(g*255), round(b*255));
			colourMap[material] = c1;
		}
	}
	mtl.close();
	// std::cout << colourMap.size() << std::endl;

    // std::ifstream inF("../src/textured-cornell-box.obj", std::ifstream::in);
	std::ifstream inF("./src/sphere.obj", std::ifstream::in);
    std::string line = "";
	std::vector<glm::vec3> verts;
	std::vector<glm::vec3> vNormal;
	std::vector<TexturePoint> vTex;
    std::vector<ModelTriangle> faces;
	Colour colour;

	while (std::getline(inF, line)) {
        if (line.empty()) continue;
		auto vector = split(line, ' ');
		if (vector[0] == "usemtl") {
			std::string dummy, material;
			std::stringstream ss(line);
			ss >> dummy >> material;
			colour = colourMap.at(material);
		}
        else if (vector[0] == "v") {
            float x, y, z;
            char dummy;
            std::stringstream ss(line);
            ss >> dummy >> x >> y >> z;
            verts.push_back(glm::vec3(scale*x, scale*y, scale*z));
        }
		else if (vector[0] == "vt") {
			float x, y;
            char dummy;
            std::stringstream ss(line);
            ss >> dummy >> x >> y;
            vTex.push_back(TexturePoint(x, y));
		}
		else if (vector[0] == "vn") {
			float x, y, z;
            char dummy;
            std::stringstream ss(line);
            ss >> dummy >> x >> y >> z;
            vNormal.push_back(glm::vec3(x, y, z));
		}
        else if (vector[0] == "f") {
            ModelTriangle m;
            for(int i = 0; i < 3; i++) {
				auto v = split(vector[i + 1], '/');
				m.vertices[i] = verts[std::stoi(v[0]) - 1];
				// for( size_t m = 0; m < v.size(); m++) {
				// 	std::cout << v[m] << '|';
				// }
				if(v.size() == 2 && !v[1].empty()) {
					// std::cout << v[1] << ' ';
					m.texturePoints[i] = vTex[std::stoi(v[1]) - 1];
					// m.texMap = TextureMap("../src/texture.ppm");
				}
				if(v.size() == 3 && v[2] != "") {
					// std::cout << v[2] << ' ';
					m.vertexNormal[i] = vNormal[std::stoi(v[2]) - 1];
				}
			}
			// std::cout << std::endl;
			m.colour = colour;
			m.normal = glm::normalize(glm::cross((m.vertices[1] - m.vertices[0]), (m.vertices[2] - m.vertices[0])));
			// std::cout << m.normal.x << std::endl;
            faces.push_back(m);
        }
        else continue;
	}
	// std::cout << "object loaded";
	inF.close();
	return faces;
}

void drawModel(float** zBuffer, DrawingWindow &window) {
	std::vector<ModelTriangle> faces = readObj(0.17);
	glm::mat4 camera(1.0, 0.0, 0.0, 0.0,
					 0.0, 1.0, 0.0, 0.0,
					 0.0, 0.0, 1.0, 0.0,
					 0.0, 0.0, 4.0, 1.0);
	float focalLength = 2;
	CanvasPoint a,b,c;
	for(size_t i = 0; i < faces.size(); i++) {
		// std::cout << 1;
		a = getCanvasIntersectionPoint(camera, faces[i].vertices[0], focalLength, 400);
		b = getCanvasIntersectionPoint(camera, faces[i].vertices[1], focalLength, 400);
		c = getCanvasIntersectionPoint(camera, faces[i].vertices[2], focalLength, 400);
		// draw normal line of each face
		// Colour white(255,255,255);
		// CanvasPoint n = getCanvasIntersectionPoint(camera, (faces[i].vertices[0] + faces[i].normal), focalLength);
		// drawLine(a, n, zBuffer, white, window);

		// std::cout << 1;
		CanvasTriangle t(a,b,c);
		// if(faces[i].texMap.width != 0) {
		// 	drawTexture(t, zBuffer, faces[i].texMap, window);
		// } else {
			// std::cout << faces[i].vertices[0].x << '|';
		drawFilledTriangle(t, faces[i].colour, zBuffer, window);	
		// }
	}
}

float getPixelColour(RayTriangleIntersection intersect, glm::vec3 light, glm::vec3 eyeDir, std::vector<ModelTriangle> &triangles, Settings &setting) {
		ModelTriangle trian = intersect.intersectedTriangle;
		float ui = intersect.intersectionPoint[1];
		float vi = intersect.intersectionPoint[2];
		glm::vec3 point = trian.vertices[0] + ui * (trian.vertices[1] - trian.vertices[0]) + vi * (trian.vertices[2] - trian.vertices[0]);
		glm::vec3 shadow_ray = light - point;
		glm::vec3 normalisedSR = glm::normalize(shadow_ray);
		glm::vec3 normal;
		if(setting.phongShader) normal = (1 - ui - vi) * trian.vertexNormal[0] + ui * trian.vertexNormal[1] + vi * trian.vertexNormal[2];
		else normal = trian.normal;

		float lightIntensity = 1.0f;
		float lightdistance = 1.0f;
		float incident = 1.0f;
		float specular = 1.0f;

		for(size_t i = 0; i < triangles.size(); i++) {
			if(intersect.triangleIndex == i) continue;
			ModelTriangle tri = triangles[i];
			// same as getClosestIntersection
			glm::vec3 e0 = tri.vertices[1] - tri.vertices[0];
			glm::vec3 e1 = tri.vertices[2] - tri.vertices[0];
			glm::vec3 sp_vector = point - tri.vertices[0];
			glm::mat3 de_matrix(-normalisedSR, e0, e1);
			glm::vec3 possible_s = inverse(de_matrix) * sp_vector;
			float t = possible_s.x, u = possible_s.y, v = possible_s.z;

			if((u >= 0.0) && (u <= 1.0) && (v >= 0.0) && (v <= 1.0) && (u + v) <= 1.0) {
				if(t < glm::length(shadow_ray) && t > 0.01) {
					return setting.ambient;
				}
			}
		}

		if(setting.proximity) lightdistance = fmin((3.0f/(4*PI*pow(glm::length(shadow_ray), 2))), 1.0f);
		if(setting.incident) incident = fmax((glm::dot(normalisedSR, normal)), 0.f);
		if(setting.specular) specular = std::pow(glm::dot(eyeDir, glm::normalize(normalisedSR - normal * 2.0f * glm::dot(normalisedSR, normal))), setting.specularStrength);
		lightIntensity = fmin(1.0f, (incident * lightdistance + setting.ambient));
		lightIntensity = fmax(lightIntensity, specular);
		return lightIntensity;
}

void drawRayTracing(DrawingWindow &window, glm::mat4 &camera, glm::vec3 &light, std::vector<ModelTriangle> &faces, Settings &setting) {
	window.clearPixels();
	// std::vector<ModelTriangle> faces = readObj(0.17);
	// You can set reflective triangle manually here
	faces[10].name = "Reflection";
	faces[11].name = "Reflection";

	// create a triangle at light position for debugging.
	// Colour white(255, 255, 255);
	// ModelTriangle lightTri((light + glm::vec3(0.025, -0.025, 0.05)), (light + glm::vec3(-0.025, -0.025, 0.05)), (light + glm::vec3(0, 0.035, 0.05)), white);
	// faces.push_back(lightTri);

	glm::vec3 cameraPos(camera[3].x, camera[3].y, camera[3].z);
	glm::mat3 cameraOri(glm::vec3(camera[0].x, camera[0].y, camera[0].z),
						glm::vec3(camera[1].x, camera[1].y, camera[1].z),
						glm::vec3(camera[2].x, camera[2].y, camera[2].z));
	
	for(int h=0; h < HEIGHT; h++) {
		for(int w=0; w < WIDTH; w++) {
			// Colour colour;
			glm::vec3 rayDirection((w - float(WIDTH/2))/500, -(h - float(HEIGHT/2))/500, -2.0f);
			glm::vec3 cameraDir = cameraOri * rayDirection;
			RayTriangleIntersection i = getClosestIntersection(cameraPos, glm::normalize(cameraDir), faces);
			Colour colour = i.intersectedTriangle.colour;
			if(i.distanceFromCamera == std::numeric_limits<float>::infinity()) continue;
			if(i.intersectedTriangle.name == "Reflection") {
				ModelTriangle trian = i.intersectedTriangle;
				float ui = i.intersectionPoint[1];
				float vi = i.intersectionPoint[2];
				glm::vec3 point = trian.vertices[0] + ui * (trian.vertices[1] - trian.vertices[0]) + vi * (trian.vertices[2] - trian.vertices[0]);
				glm::vec3 normal = (1 - ui - vi) * trian.vertexNormal[0] + ui * trian.vertexNormal[1] + vi * trian.vertexNormal[2];
				if (normal[0] == 0 && normal[1] == 0 && normal[2] == 0) normal = trian.normal;
				auto newCamDir = glm::normalize(cameraDir - 2.0f * normal * glm::dot(cameraDir, normal));
				i = getClosestIntersection(point, newCamDir, faces);
				colour = i.intersectedTriangle.colour;
			}

			if(setting.texture == true && (i.triangleIndex == 6 || i.triangleIndex == 7)) {
				TextureMap texture("./src/texture.ppm");
				auto w = texture.width;
				auto h = texture.height;
				float uj = i.intersectionPoint[1];
				float vj = i.intersectionPoint[2];
				int x = int((1 - uj - vj) * i.intersectedTriangle.texturePoints[0].x * w + uj * i.intersectedTriangle.texturePoints[1].x * w + vj * i.intersectedTriangle.texturePoints[2].x * w) % w;
				int y = int((1 - uj - vj) * i.intersectedTriangle.texturePoints[0].y * h + uj * i.intersectedTriangle.texturePoints[1].y * h + vj * i.intersectedTriangle.texturePoints[2].y * h) % h;
				uint32_t texColour = texture.pixels[w * y + x];
				colour = Colour(texColour >> 16 & 0xFF, texColour >> 8 & 0xFF, texColour & 0xFF);
			}

			std::vector<glm::vec3> lights;
			lights.push_back(light);
			if(!setting.softShadow) {
				float intensity = getPixelColour(i, light, glm::normalize(cameraDir), faces, setting);
				uint32_t c = (255 << 24) + (int(colour.red*intensity) << 16) + (int(colour.green*intensity) << 8) + int(colour.blue*intensity);	
				window.setPixelColour(w, h, c);
			} else {
				for(int i=0; i<2; i++) {
					auto j = (i + 1) * 0.025f;
					lights.push_back(light + glm::vec3(-j, 0, j));
					lights.push_back(light + glm::vec3(-j, 0, 0));
					lights.push_back(light + glm::vec3(-j, 0, -j));
					lights.push_back(light + glm::vec3(0, 0, -j));
					lights.push_back(light + glm::vec3(0, 0, j));
					lights.push_back(light + glm::vec3(j, 0, j));
					lights.push_back(light + glm::vec3(j, 0, 0));
					lights.push_back(light + glm::vec3(j, 0, -j));
				}
				float acc=0.0f;
				for(size_t j = 0; j < lights.size(); j++) {
					float intensity = getPixelColour(i, lights[j], glm::normalize(cameraDir), faces, setting);
					acc += intensity;
				}
				uint32_t c = (255 << 24) + (int(colour.red*acc/17.0f) << 16) + (int(colour.green*acc/17.0f) << 8) + int(colour.blue*acc/17.0f);	
				window.setPixelColour(w, h, c);
			}
		}
	}
	std::cout << "rayTracing finish" << std::endl;
}

CanvasTriangle randomTriangle() {
	CanvasPoint a, b, c;
	CanvasTriangle ran(a, b, c);
	for (int i = 0; i < 3; i++)
	{
		ran[i].x = float(rand()%WIDTH);
		ran[i].y = float(rand()%HEIGHT);
	}
	return ran;
}

Colour randomColour() {
	int red = int(rand()%256);
	int green = int(rand()%256);
	int blue = int(rand()%256);
	Colour c(red, green, blue);
	return c;
}

void drawM(DrawingWindow &window, glm::mat4 camera, std::vector<ModelTriangle> &faces, int &mode) {
	window.clearPixels();
	float** zBuffer = new float*[HEIGHT];
	for(int i = 0; i < HEIGHT; i++) { zBuffer[i] = new float[WIDTH]; }
	for(int h = 0; h < HEIGHT; h++) {
		for(int w = 0; w < WIDTH; w++) {
			zBuffer[h][w] = 0;
		}
	}

	float focalLength = 2;
	CanvasPoint a,b,c;
	for(size_t i = 0; i < faces.size(); i++) {
		a = getCanvasIntersectionPoint(camera, faces[i].vertices[0], focalLength, 240);
		b = getCanvasIntersectionPoint(camera, faces[i].vertices[1], focalLength, 240);
		c = getCanvasIntersectionPoint(camera, faces[i].vertices[2], focalLength, 240);
		CanvasTriangle t(a,b,c);
		if(mode == 1) {
			drawFilledTriangle(t, faces[i].colour, zBuffer, window);
		} else if(mode == 3) {
			drawStrokedTriangle(t, zBuffer, faces[i].colour, window);
		}
		
	}
	// std::vector<float> depthBuffer = std::vector<float>(window.height * window.width, 0);

	for(int i = 0; i < HEIGHT; i++) { delete []zBuffer[i]; }
	delete []zBuffer;
}

void handleEvent(SDL_Event event, DrawingWindow &window, glm::mat4 &camera, glm::vec3 &light, int &mode, Settings &setting, bool &draw, std::vector<ModelTriangle> &faces) {
	if (event.type == SDL_KEYDOWN) {
		if (event.key.keysym.sym == SDLK_LEFT) camera = translate_matrix(0.1f, 0.0f, 0.0f) * camera;
		else if (event.key.keysym.sym == SDLK_RIGHT) camera = translate_matrix(-0.1f, 0.0f, 0.0f) * camera;
		else if (event.key.keysym.sym == SDLK_UP) camera = translate_matrix(0.0f, -0.1f, 0.0f) * camera;
		else if (event.key.keysym.sym == SDLK_DOWN) camera = translate_matrix(0.0f, 0.1f, 0.0f) * camera;
		else if (event.key.keysym.sym == '.') camera = translate_matrix(0.0f, 0.0f, 0.1f) * camera;
		else if (event.key.keysym.sym == '/') camera = translate_matrix(0.0f, 0.0f, -0.1f) * camera;
		else if (event.key.keysym.sym == 'w') camera = rotate_matrix(3.0f, 0.0f, 0.0f) * camera;
		else if (event.key.keysym.sym == 's') camera = rotate_matrix(-3.0f, 0.0f, 0.0f) * camera;
		else if (event.key.keysym.sym == 'a') camera = rotate_matrix(0.0f, 3.0f, 0.0f) * camera;
		else if (event.key.keysym.sym == 'd') camera = rotate_matrix(0.0f, -3.0f, 0.0f) * camera;
		else if (event.key.keysym.sym == 'q') camera = rotate_matrix(0.0f, 0.0f, 3.0f) * camera;
		else if (event.key.keysym.sym == 'e') camera = rotate_matrix(0.0f, 0.0f, -3.0f) * camera;
		else if (event.key.keysym.sym == '0') camera = lookAt(camera, glm::vec3(0.f));
		else if (event.key.keysym.sym == 'h') draw = !draw;
		else if (event.key.keysym.sym == 'k') light = light + glm::vec3(0.0f, -0.05f, 0.0f);
		else if (event.key.keysym.sym == 'i') light = light + glm::vec3(0.0f, 0.05f, 0.0f);
		else if (event.key.keysym.sym == 'j') light = light + glm::vec3(-0.05f, 0.0f, 0.0f);
		else if (event.key.keysym.sym == 'l') light = light + glm::vec3(0.05f, 0.0f, 0.0f);
		else if (event.key.keysym.sym == '[') light = light + glm::vec3(0.0f, 0.0f, 0.05f);
		else if (event.key.keysym.sym == ']') light = light + glm::vec3(0.0f, 0.0f, -0.05f);
		else if (event.key.keysym.sym == '1') { mode = 1; std::cout << "rasterize mode" << std::endl; }
		else if (event.key.keysym.sym == '2') { mode = 2; std::cout << "ray tracing mode" << std::endl; }
		else if (event.key.keysym.sym == '3') { mode = 3; std::cout << "wireframe mode" << std::endl; }
		else if (event.key.keysym.sym == '4') { setting.proximity = !setting.proximity; std::cout << "proximity switched" << std::endl; }
		else if (event.key.keysym.sym == '5') { setting.incident = !setting.incident; std::cout << "incident switched" << std::endl; }
		else if (event.key.keysym.sym == '6') { setting.specular = !setting.specular; std::cout << "specular switched" << std::endl; }
		else if (event.key.keysym.sym == '7') { setting.phongShader = !setting.phongShader; std::cout << "phong switched" << std::endl; }
		else if (event.key.keysym.sym == '9') { setting.softShadow = !setting.softShadow; std::cout<< "soft shadow switch" << std::endl; }
		// 'u' for random stroked triangles, 'f' for random filled triangles.
		else if (event.key.keysym.sym == 'u') {
			float** zBuffer = new float*[HEIGHT];
			for(int i = 0; i < HEIGHT; i++) { zBuffer[i] = new float[WIDTH]; }
			for(int h = 0; h < HEIGHT; h++) {
				for(int w = 0; w < WIDTH; w++) {
					zBuffer[h][w] = 0;
				}
			}

			CanvasTriangle triangle = randomTriangle();
			Colour colour = randomColour();
			drawStrokedTriangle(triangle, zBuffer, colour, window);

			for(int i = 0; i < HEIGHT; i++) { delete []zBuffer[i]; }
			delete []zBuffer;
		}else if (event.key.keysym.sym == 'f') {
			float** zBuffer = new float*[HEIGHT];
			for(int i = 0; i < HEIGHT; i++) { zBuffer[i] = new float[WIDTH]; }
			for(int h = 0; h < HEIGHT; h++) {
				for(int w = 0; w < WIDTH; w++) {
					zBuffer[h][w] = 0;
				}
			}

			CanvasTriangle triangle = randomTriangle();
			Colour colour = randomColour();
			drawFilledTriangle(triangle, colour, zBuffer, window);

			for(int i = 0; i < HEIGHT; i++) { delete []zBuffer[i]; }
			delete []zBuffer;
		}else if (event.key.keysym.sym == 't') {
			float** zBuffer = new float*[HEIGHT];
			for(int i = 0; i < HEIGHT; i++) { zBuffer[i] = new float[WIDTH]; }
			for(int h = 0; h < HEIGHT; h++) {
				for(int w = 0; w < WIDTH; w++) {
					zBuffer[h][w] = 0;
				}
			}
			CanvasPoint a(160, 10), b(10, 150), c(300, 230);
			CanvasTriangle triangle(a, b, c);
			triangle[0].texturePoint = TexturePoint(195, 5);
			triangle[1].texturePoint = TexturePoint(65, 330);
			triangle[2].texturePoint = TexturePoint(395, 380);
			TextureMap texture("./src/texture.ppm");
			drawTexture(triangle, zBuffer, texture, window);

			for(int i = 0; i < HEIGHT; i++) { delete []zBuffer[i]; }
			delete []zBuffer;
		}else if (event.key.keysym.sym == 'm') {
			drawM(window, camera, faces, mode);
		} else if (event.key.keysym.sym == 'r') {
		 	drawRayTracing(window, camera, light, faces, setting);
		}

	} else if (event.type == SDL_MOUSEBUTTONDOWN) {
		window.savePPM("output.ppm");
		window.saveBMP("output.bmp");
	}
}


int main(int argc, char *argv[]) {
	DrawingWindow window = DrawingWindow(WIDTH, HEIGHT, false);
	SDL_Event event;
	glm::mat4 camera(1.0, 0.0, 0.0, 0.0,
					 0.0, 1.0, 0.0, 0.0,
					 0.0, 0.0, 1.0, 0.0,
					 0.0, 0.3, 4.0, 1.0);
	glm::vec3 light(0.15f, 0.25f, 0.5f);
	std::vector<ModelTriangle> faces = readObj(0.17);
	// faces[6].texturePoints = {TexturePoint(0.9, 0.9), TexturePoint(0.1, 0.1), TexturePoint(0.1, 0.9)};
	// faces[7].texturePoints = {TexturePoint(0.9, 0.9), TexturePoint(0.9, 0.1), TexturePoint(0.1, 0.1)};
	// glm::vec3 light(0.0f, 0.4f, 0.0f);
	int mode = 2;
	bool draw = false;
	Settings setting(true, true, true, false, true, false, 256, 0.2);	//bool proximity, bool incident, bool specular, bool texture, bool phongShader, bool softShadow, float specularStrength, float ambient
	int num = 0;
	while (true) {
		// We MUST poll for events - otherwise the window will freeze !
		if (window.pollForInputEvents(event)) handleEvent(event, window, camera, light, mode, setting, draw, faces);
		if (draw) {
			if (mode == 2) {
				drawRayTracing(window, camera, light, faces, setting);
				// camera = rotate_matrix(0.0f, -2.0f, 0.0f) * camera;
				light = light + glm::vec3(-0.05f, 0.0f, 0.0f);
				// char buf[10];
				// std::sprintf(buf, "%03d", num);
				// std::string buff = buf;
				// window.savePPM("sphere/pic" + buff + ".ppm");
				num++;
			} else {
				drawM(window, camera, faces, mode);
				camera = rotate_matrix(0.0f, 2.0f, 0.0f) * camera;
				char buf[10];
				std::sprintf(buf, "%03d", num);
				std::string buff = buf;
				window.savePPM("pic/pic" + buff + ".ppm");
				num++;
			}
		}

		// Need to render the frame at the end, or nothing actually gets shown on the screen !
		window.renderFrame();
	}
}

// github token: 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 	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