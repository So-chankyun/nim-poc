import time
import asyncio
import aiohttp
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import pandas as pd
import random


@dataclass
class BenchmarkResult:
    ttft: float  # Time to First Token (ms)
    e2e_latency: float  # End-to-end latency (ms) 
    itl: float  # Inter-token latency (ms)
    tps_per_user: float  # Tokens per second per user
    input_tokens: int
    output_tokens: int

# @dataclass
# class QASet:
#     prompt: string
#     answer: string
    
class LLMBenchmark:
    def __init__(self, api_endpoint: str, headers: Dict[str, str], model):
        self.api_endpoint = api_endpoint
        self.headers = headers
        self.model = model
        self.kst = ZoneInfo("Asia/Seoul")
    
    def res_json_parsing(self,data:str) -> Dict:
        data
        
    async def single_request_benchmark(self, prompt: str, max_tokens: int = 100) -> BenchmarkResult:
        """단일 요청의 성능 지표 측정"""
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,  # 스트리밍 모드로 TTFT 측정
            "temperature": 0,  # 일관된 결과를 위해
        }
        
        start_time = time.time()
        first_token_time = None
        token_times = []
        tokens_received = 0
        final_usage = None
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_endpoint, 
                headers=self.headers, 
                json=payload
            ) as response:
                
                # 응답 json 파싱
                async for line in response.content:
                    if line:
                        current_time = time.time()
                        
                        # 첫 토큰 수신 시간 기록
                        if first_token_time is None:
                            first_token_time = current_time
                            
                        # 토큰별 시간 기록
                        token_times.append(current_time)
                        tokens_received += 1
        
        end_time = time.time()
        
        # 메트릭 계산
        ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
        e2e_latency = (end_time - start_time) * 1000
        
        # ITL 계산 (첫 토큰 제외)
        if len(token_times) > 1:
            inter_token_intervals = [
                (token_times[i] - token_times[i-1]) * 1000 
                for i in range(1, len(token_times))
            ]
            itl = statistics.mean(inter_token_intervals)
        else:
            itl = 0
            
        # TPS per user 계산
        generation_time = (end_time - first_token_time) if first_token_time else e2e_latency / 1000
        tps_per_user = tokens_received / generation_time if generation_time > 0 else 0
        
        return BenchmarkResult(
            ttft=ttft,
            e2e_latency=e2e_latency,
            itl=itl,
            tps_per_user=tps_per_user,
            input_tokens=len(prompt.split()),  # 근사치
            output_tokens=tokens_received
        )
    
    async def concurrent_benchmark(self, prompt: str, concurrency: int, max_tokens: int = 100) -> Dict[str, Any]:
        """동시성 테스트로 시스템 처리량 측정"""
        
        # 동시 요청 실행
        benchmark_start = time.time()
        tasks = [
            self.single_request_benchmark(prompt[i]["question"], max_tokens) 
            for i in range(concurrency)
        ]
        results = await asyncio.gather(*tasks)
        benchmark_end = time.time()
        
        # 시스템 전체 TPS 계산
        total_output_tokens = sum(r.output_tokens for r in results)
        total_benchmark_time = benchmark_end - benchmark_start
        system_tps = total_output_tokens / total_benchmark_time
        
        # RPS 계산
        rps = concurrency / total_benchmark_time
        
        # 통계 계산
        ttfts = [r.ttft for r in results]
        e2e_latencies = [r.e2e_latency for r in results]
        itls = [r.itl for r in results if r.itl > 0] # 각 요청의 평균 ITL 을 반환
        
        return {
            "concurrency": concurrency,
            "total_requests": len(results),
            "system_tps": system_tps,
            "rps": rps,
            "ttft_stats": {
                "mean": statistics.mean(ttfts),
                "median": statistics.median(ttfts)
                # "p95": statistics.quantiles(ttfts, n=20)[18],  # 95th percentile
                # "p99": statistics.quantiles(ttfts, n=100)[98]  # 99th percentile
            },
            "e2e_latency_stats": {
                "mean": statistics.mean(e2e_latencies),
                "median": statistics.median(e2e_latencies)
                # "p95": statistics.quantiles(e2e_latencies, n=20)[18],
                # "p99": statistics.quantiles(e2e_latencies, n=100)[98]
            },
            "itl_stats": {
                "mean": statistics.mean(itls) if itls else 0,
                "median": statistics.median(itls) if itls else 0
            },
            "total_output_tokens": total_output_tokens,
            "benchmark_duration": total_benchmark_time
        }
    
    async def load_test(self, prompt: str, concurrency_levels: List[int], max_tokens: int = 100) -> List[Dict[str, Any]]:
        """다양한 동시성 레벨에서 부하 테스트"""
        results = []
        
        for concurrency in concurrency_levels:
            print(f"Testing concurrency level: {concurrency}")
            result = await self.concurrent_benchmark(prompt, concurrency, max_tokens)
            results.append(result)
            
            # 서버 안정화를 위한 대기
            await asyncio.sleep(1)
        
        return results

# 사용 예제
async def main():
    # Set timezone
    kst = ZoneInfo("Asia/Seoul")

    # API 설정 (실제 엔드포인트와 헤더로 교체 필요)
    api_endpoint = "http://10.150.109.164:8000/v1/chat/completions"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    
    # model = "meta/llama-3.1-8b-instruct" # nim
    # model = "meta-llama/Llama-3.1-8B-Instruct" # vllm
    model = "openai/gpt-oss-120b"
    
    benchmark = LLMBenchmark(api_endpoint, headers, model)
    
    # 테스트 프롬프트
    # 1. Read prompt
    prompts = None
    
    with open("./input_prompt.json",'r',encoding='utf-8') as f:
        prompts = json.load(f)
    
    # prompt = "Explain the benefits of renewable energy in detail."
    
    # print("=== 단일 요청 벤치마크 ===")
    # single_result = await benchmark.single_request_benchmark(prompts[0]["question"], max_tokens=8192)
    # print(f"TTFT: {single_result.ttft:.2f}ms")
    # print(f"E2E Latency: {single_result.e2e_latency:.2f}ms")
    # print(f"ITL: {single_result.itl:.2f}ms")
    # print(f"TPS per user: {single_result.tps_per_user:.2f}")
    # print(f"Output tokens: {single_result.output_tokens}")
    
    # 50회 반복하고, 결과를 저장
    
    data = []
    
    start_time = time.time()
    
    for i in range(0,1):
      # shuffle prompts
      shuffled_prompts = random.sample(prompts,len(prompts))
      
      print(f"\n=== {i+1}차 부하 테스트 ===")
      concurrency_levels = [1, 5, 10, 20]
      load_results = await benchmark.load_test(shuffled_prompts, concurrency_levels, max_tokens=8192)
      
      for result in load_results:
          print(f"\nConcurrency {result['concurrency']}:")
          print(f"  System TPS: {result['system_tps']:.2f}")
          print(f"  RPS: {result['rps']:.2f}")
          # print(f"  TTFT P95: {result['ttft_stats']['p95']:.2f}ms")
          print(f"  TTFT mean: {result['ttft_stats']['mean']:.2f}ms")
          # print(f"  E2E Latency P95: {result['e2e_latency_stats']['p95']:.2f}ms")
          print(f"  E2E Latency mean: {result['e2e_latency_stats']['mean']:.2f}ms")
          print(f"  ITL mean: {result['itl_stats']['mean']:.2f}ms")
          print(f"  Total Output Token: {result['total_output_tokens']}")
          print(f"  Benchmark Duration: {result['benchmark_duration']:.2f}s")    
      
          data.append({"timestamp":datetime.now(tz=kst)
                      ,"concurrency_level":result['concurrency']
                      ,"system_tps":round(result['system_tps'],2)
                      ,"rps":round(result['rps'],2)
                      ,"ttft_mean":round(result['ttft_stats']['mean'],2)
                      ,"e2e_latency_mean":round(result['e2e_latency_stats']['mean'],2)
                      ,"itl_mean":round(result['itl_stats']['mean'],2)
                      ,"benchmark_duration":round(result['benchmark_duration'],2)
                      })
    end_time = time.time()
    
    test_duration = round(end_time - start_time,2)
    
    print(f"Total Test Duration Time : {test_duration}s")
    
    df = pd.DataFrame(data,columns=[
      "timestamp",
      "concurrency_level",
      "system_tps",
      "rps",
      "ttft_mean",
      "e2e_latency_mean",
      "itl_mean",
      "benchmark_duration"
    ])
    
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    model_name = model.split("/")[1]
    
    output_path = "./results"
    # 1. nim model
    if model_name == "llama-3.1-8b-instruct":
      output_path += f"/nim_{datetime.now(tz=kst)}.csv"
    # 2. vllm model
    else:
      output_path += f"/vllm_{datetime.now(tz=kst)}.csv"
      
    df.to_csv(output_path,index=False, encoding="utf-8")
    print(f"Saved to {output_path}")    
    

# 실행
if __name__ == "__main__":
    asyncio.run(main())