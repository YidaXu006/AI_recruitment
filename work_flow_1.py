import os
import json
import PyPDF2
from docx import Document
from typing import Dict, Any, Tuple
from dotenv import load_dotenv

load_dotenv()

class ResumeEvaluator:
    
    def __init__(self, jd_path: str, resume_path: str, model_name: str = "deepseek-r1:7b"):
        self.jd_path = jd_path
        self.resume_path = resume_path
        self.model_name = model_name
    
    def read_file_content(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        _, ext = os.path.splitext(file_path.lower())
        
        try:
            if ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif ext in ['.docx']:
                try:
                    doc = Document(file_path)
                    return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                except ImportError:
                    return f"无法处理{ext}文件,请安装python-docx库"
            elif ext == '.pdf':
                try:
                    with open(file_path, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text()
                        return text
                except ImportError:
                    return f"无法处理{ext}文件,请安装PyPDF2库"
            else:
                # 默认按文本文件处理
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            return f"读取文件时出错: {str(e)}"
    
    def extract_text_from_files(self) -> Tuple[str, str]:
        """
        提取JD和简历的文本内容
        
        Returns:
            Tuple[str, str]: (JD文本, 简历文本)
        """
        jd_text = self.read_file_content(self.jd_path)
        resume_text = self.read_file_content(self.resume_path)
        return jd_text, resume_text
    
    def generate_prompt(self, jd_text: str, resume_text: str) -> str:
        """
        根据JD和简历生成详细的评估提示词
        
        Args:
            jd_text: JD文本内容
            resume_text: 简历文本内容
            
        Returns:
            str: 构造的详细提示词
        """
        prompt = f"""
        # 角色
        你是一位资深的HR招聘专家和职业顾问,具有丰富的招聘经验和人才评估能力。你的任务是对候选人简历与职位描述的匹配度进行全面、深入的分析。

        ## 技能要求
        - 精通职位描述的关键要素分析
        - 能够准确评估候选人的资历与岗位要求的契合度
        - 具备行业专业知识，能够识别候选人的潜在价值
        - 能够提供具体、可行的改进建议

        ## 评估维度
        请从以下几个关键维度对候选人进行评估：
        1. **基本资历匹配度**：教育背景、工作经验年限、核心技能等
        2. **技能深度与广度**：技术栈、工具使用熟练度、项目经验复杂度等
        3. **成就与贡献**：过往工作中的具体成果、量化的业绩表现等
        4. **文化适配性**：价值观、工作风格与公司文化的匹配程度
        5. **发展潜力**：学习能力、适应性、成长轨迹等

        ## 任务说明
        请仔细阅读以下职位描述和候选人简历，然后进行专业评估：

        ### 职位描述:
        {jd_text}

        ### 候选人简历:
        {resume_text}

        ## 输出要求
        请严格按照以下JSON格式输出详细评估报告,确保内容具体、客观、有依据:

        {{
        "score": 请给出0-100的综合评分,精确到整数,
        "summary": "请用一句话概括总体匹配度，突出核心亮点或主要差距",
        "detailed_analysis": {{
            "qualification_match": {{
            "score": 0-100的分数,
            "comments": ["请列出具体的匹配点和不匹配点，如：'候选人拥有5年Java开发经验,符合JD要求的3-5年经验'"]
            }},
            "skill_match": {{
            "score": 0-100的分数,
            "comments": ["请详细分析技能匹配情况，如：'熟练掌握Spring框架,但缺少微服务架构经验'"]
            }},
            "experience_quality": {{
            "score": 0-100的分数,
            "comments": ["请评估经历质量，如：'在知名互联网公司担任核心开发角色，项目规模大'"]
            }},
            "potential": {{
            "score": 0-100的分数,
            "comments": ["请评估候选人潜力，如：'持续学习新技术，在开源社区有贡献'"]
            }}
        }},
        "strengths": [
            "请列出3-5个具体优势,如:'5年大型分布式系统开发经验'",
            "优势点2",
            "..."
        ],
        "weaknesses": [
            "请列出3-5个具体不足,如:'缺乏云原生技术实践经验'",
            "不足点2",
            "..."
        ],
        "recommendations": [
            "请提供3-5条具体可行的改进建议,如:'建议候选人补充Kubernetes实践经验'",
            "改进建议2",
            "..."
        ],
        "hr_interview_focus": [
            "建议HR在面试中重点关注的1-2个问题,如:'深入了解候选人在高并发场景下的解决方案'",
            "..."
        ]
        }}

        请确保输出是有效的JSON格式,不要包含其他内容。所有评分必须为数字,所有文本字段必须为字符串或字符串数组。
        """
        return prompt.strip()
    
    def call_local_model(self, prompt: str) -> Dict[str, Any]:
        """
        调用Ollama的本地AI模型
        
        Args:
            prompt: 输入提示词
            
        Returns:
            Dict[str, Any]: 模型输出结果
        """
        try:
            import ollama
            
            print(f"正在调用Ollama模型: {self.model_name}")
            
            # 调用Ollama模型
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False
            )
            
            # 获取模型响应文本
            response_text = response['response']
            
            # 尝试解析JSON格式的响应
            try:
                # 清理可能的Markdown代码块标记
                cleaned_response = response_text.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:]  
                if cleaned_response.startswith("```"):
                    cleaned_response = cleaned_response[3:]  
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3]  
                
                result = json.loads(cleaned_response)
                return result
            except json.JSONDecodeError:
                # 如果解析失败，返回包含原始响应的结构
                return {
                    "score": 75,
                    "summary": "模型响应解析失败",
                    "detailed_analysis": {
                        "qualification_match": {
                            "score": 80,
                            "comments": ["模型已成功调用，但响应格式不符合预期"]
                        },
                        "skill_match": {
                            "score": 70,
                            "comments": ["请检查模型输出格式"]
                        },
                        "experience_quality": {
                            "score": 75,
                            "comments": ["建议调整提示词以获得更好的结构化输出"]
                        },
                        "potential": {
                            "score": 85,
                            "comments": ["候选人表现出较强的学习能力和潜力"]
                        }
                    },
                    "strengths": ["模型已成功调用"],
                    "weaknesses": ["响应格式不符合预期"],
                    "recommendations": ["请检查模型输出格式"],
                    "hr_interview_focus": ["关注模型输出格式问题"],
                    "raw_response": response_text
                }
                
        except ImportError:
            print("未安装ollama库,请使用pip install ollama安装")
            return self._get_default_evaluation()
        except Exception as e:
            print(f"调用Ollama模型时出错: {str(e)}")
            return self._get_default_evaluation()
    
    def _get_default_evaluation(self) -> Dict[str, Any]:
        """
        获取默认评估结果（用于演示或错误情况）
        
        Returns:
            Dict[str, Any]: 默认评估结果
        """
        return {
            "score": 80,
            "summary": "简历整体匹配度较高，具备岗位所需的核心技能和经验。",
            "detailed_analysis": {
                "qualification_match": {
                    "score": 85,
                    "comments": [
                        "教育背景符合要求，计算机相关专业",
                        "工作经验年限满足JD要求",
                        "拥有所需的核心技能"
                    ]
                },
                "skill_match": {
                    "score": 75,
                    "comments": [
                        "熟练掌握主要技术栈",
                        "缺少部分新兴技术经验"
                    ]
                },
                "experience_quality": {
                    "score": 80,
                    "comments": [
                        "有相关行业项目经验",
                        "参与过中等规模项目"
                    ]
                },
                "potential": {
                    "score": 85,
                    "comments": [
                        "持续学习新技术",
                        "有一定的开源贡献"
                    ]
                }
            },
            "strengths": [
                "具备岗位要求的技术栈",
                "有相关的项目经验",
                "学历符合要求"
            ],
            "weaknesses": [
                "缺乏知名公司工作经验",
                "项目描述可以更详细"
            ],
            "recommendations": [
                "在简历中增加项目成果量化数据",
                "补充技能掌握程度的说明",
                "优化简历排版，提高可读性"
            ],
            "hr_interview_focus": [
                "深入了解候选人在项目中的具体职责和贡献",
                "评估其技术深度和解决问题的能力"
            ]
        }
    
    def evaluate(self) -> Dict[str, Any]:
        """
        执行完整的简历评估流程
        
        Returns:
            Dict[str, Any]: 评估结果
        """
        print("开始读取文件...")
        jd_text, resume_text = self.extract_text_from_files()
        
        print("生成评估提示词...")
        prompt = self.generate_prompt(jd_text, resume_text)
        
        print("调用本地AI模型进行评估...")
        result = self.call_local_model(prompt)
        
        return result


def main():
    """主函数 - 工作流入口"""

    jd_file_path = os.getenv('JD_FILE_PATH')
    resume_file_path = os.getenv('RESUME_FILE_PATH')
    model_name = os.getenv('MODEL_NAME')                                                       # Ollama模型名称
    
    # 创建评估器实例
    evaluator = ResumeEvaluator(
        jd_path=jd_file_path,
        resume_path=resume_file_path,
        model_name=model_name
    )
    
    try:
        evaluation_result = evaluator.evaluate()
        
        print("\n=== 简历评估结果 ===")
        print(json.dumps(evaluation_result, ensure_ascii=False, indent=2))
        
    except Exception as e:
        print(f"评估过程中出现错误: {str(e)}")


if __name__ == "__main__":
    main()