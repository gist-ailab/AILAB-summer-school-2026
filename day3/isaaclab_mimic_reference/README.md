# isaaclab_mimic Reference Snapshot

이 폴더는 IsaacLab v2.3.2의 `isaaclab_mimic` 내부 구현을 강의 중 확인하기 위한 참고용 스냅샷입니다. 실제 실습 실행은 설치된 `$ISAACLAB_PATH/source/isaaclab_mimic` 라이브러리를 사용합니다.

핵심 흐름:

```text
HDF5 obs/datagen_info
  -> DatagenInfo
  -> DataGenInfoPool에서 subtask boundary 계산
  -> DataGenerator에서 object_ref 기준 pose transform
  -> WaypointTrajectory 생성
  -> env.target_eef_pose_to_action()
  -> IsaacLab rollout
```

볼 파일:

```text
datagen_info.py       DatagenInfo 구조
datagen_info_pool.py  subtask signal의 0->1 transition으로 boundary 계산
data_generator.py     object pose 기준 EEF pose transform
selection_strategy.py source segment 선택
waypoint.py           target pose sequence 실행
```

object-centric transform을 직접 실행해 보는 축약 실습은 상위 폴더의 문제 3 파일을 사용합니다.
